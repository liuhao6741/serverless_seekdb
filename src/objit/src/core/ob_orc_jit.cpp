/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define USING_LOG_PREFIX PL
#include "core/ob_orc_jit.h"

#include <atomic>
#include <mutex>
#include <vector>

#include "llvm/Config/llvm-config.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "core/ob_pl_ir_compiler.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"

#ifdef _WIN32
#include <windows.h>
#include <stdio.h>
#include <cstdarg>
#endif

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::object;
using namespace ::oceanbase::common;

namespace oceanbase
{
namespace jit
{
namespace core
{

DenseMap<StringRef, JITTargetAddress> ObJitGlobalSymbolGenerator::symbol_table;
std::vector<std::string*> ObJitGlobalSymbolGenerator::persistent_strings;

std::pair<lib::ObMutex, ObNotifyLoaded::KeyEntryMap> ObNotifyLoaded::AllGdbReg;

namespace detail {
static std::atomic<uint64_t> g_shim_seq{0};
#ifdef _WIN32

static void ob_shim_trace(const char *fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  int n = _vsnprintf_s(buf, sizeof(buf), _TRUNCATE, fmt, ap);
  va_end(ap);
  if (n <= 0) return;
  OutputDebugStringA(buf);
}
#else
static inline void ob_shim_trace(const char *, ...) {}
#endif

class ObJitMemoryManagerShim final : public llvm::RTDyldMemoryManager
{
public:
  // Magic values: live shims have ALIVE; ~Shim flips to DEAD before returning,
  // so any virtual call after destruction is detectable.
  static constexpr uint64_t MAGIC_ALIVE = 0xA11CEDEADBEEF42AULL;
  static constexpr uint64_t MAGIC_DEAD  = 0xDEADDEADDEADDEADULL;

  explicit ObJitMemoryManagerShim(ObJitMemoryManager *delegate)
    : magic_(MAGIC_ALIVE),
      seq_(g_shim_seq.fetch_add(1, std::memory_order_relaxed) + 1),
      delegate_(delegate)
  {
  }

  ~ObJitMemoryManagerShim() override {
    // Flip magic AFTER trace so any racing virtual caller still sees Alive
    // until the moment we record the dtor. After this store, downstream
    // checks in registerEHFrames/finalizeMemory will catch use-after-dtor.
    magic_ = MAGIC_DEAD;
  }
  static void operator delete(void *) noexcept { /* shim is pinned, never freed */ }
  static void operator delete[](void *) noexcept { /* not used, but keep symmetric */ }

  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Align,
                               unsigned SectionID,
                               llvm::StringRef SectionName) override
  {
    return delegate_->allocateCodeSection(Size, Align, SectionID, SectionName);
  }

  uint8_t *allocateDataSection(uintptr_t Size, unsigned Align,
                               unsigned SectionID,
                               llvm::StringRef SectionName,
                               bool IsReadOnly) override
  {
    return delegate_->allocateDataSection(Size, Align, SectionID, SectionName,
                                          IsReadOnly);
  }

  bool finalizeMemory(std::string *ErrMsg = nullptr) override
  {
    // finalizeMemory is private in ObJitMemoryManager; route through the
    // base-class virtual to satisfy access checks while keeping virtual dispatch.
    return static_cast<llvm::RTDyldMemoryManager *>(delegate_)->finalizeMemory(ErrMsg);
  }

  void registerEHFrames(uint8_t *Addr, uint64_t LoadAddr, size_t Size) override
  {
    delegate_->registerEHFrames(Addr, LoadAddr, Size);
  }

  void deregisterEHFrames() override
  {
    delegate_->deregisterEHFrames();
  }

private:
  // Diagnostic: MAGIC_ALIVE while constructed, MAGIC_DEAD after ~Shim().
  // Placed first so its offset is stable; checked at every virtual entry.
  uint64_t magic_;
  uint64_t seq_;                  // monotonic id for matching ctor/dtor in logs
  ObJitMemoryManager *delegate_;  // pinned in persistent pool, not owning
  DISALLOW_COPY_AND_ASSIGN(ObJitMemoryManagerShim);
};

class ObJitMemMgrPool
{
public:
  ObJitMemMgrPool() {}
  ~ObJitMemMgrPool() {}
  std::mutex &get_mutex() { return mtx_; }
  std::vector<std::unique_ptr<ObJitMemoryManager>> &get_mgrs() { return mgrs_; }
private:
  std::mutex mtx_;
  std::vector<std::unique_ptr<ObJitMemoryManager>> mgrs_;
  DISALLOW_COPY_AND_ASSIGN(ObJitMemMgrPool);
};

static ObJitMemMgrPool &get_mem_mgr_pool() {
  static ObJitMemMgrPool pool;
  return pool;
}
} // namespace detail

ObOrcJit::ObOrcJit(common::ObIAllocator &Allocator)
  : DebugBuf(nullptr),
    DebugLen(0),
    JITAllocator(),
    NotifyLoaded(Allocator, DebugBuf, DebugLen, SoObject),
    ObTM(EngineBuilder().selectTarget()),
    ObDL(ObTM->createDataLayout()),
    ObEngineBuilder(),
    ObJitEngine()
{ }

int ObOrcJit::init()
{
  int ret = OB_SUCCESS;

    // NB: capture stable pointers (`alloc_ptr`, `notify_ptr`) into the inner
    // factory lambda by VALUE rather than reusing the enclosing lambda's
    // `this` via `[&]`. The outer lambda is owned by `ObEngineBuilder` and
    // can be moved-from when `LLJITBuilder::create()` consumes the builder,
    // leaving any reference into the outer closure dangling. The inner
    // factory lambda is stored inside `RTDyldObjectLinkingLayer` and gets
    // invoked once per emit() (including async re-entries triggered by
    // nested compilations such as cursor SQL).
    //
    // In addition, what LLVM gets back from the factory is now an
    // `ObJitMemoryManagerShim` rather than the real `ObJitMemoryManager`.
    // The shim is a thin forwarding object that LLVM owns and may destroy at
    // any time during ORC v2 async materialization; the real manager lives
    // in the persistent pool keyed by `get_mem_mgr_pool()` and survives the
    // shim. This decoupling fixes the Windows crash in
    // `RuntimeDyldImpl::finalizeAsync` -> `registerEHFrames` where the
    // MemMgr pointer had been freed and reused by ORC for a
    // `_Ref_count_obj2<AsynchronousSymbolQuery>` (vtable slot ended up at
    // 0xfffffffffffffff8). With the shim, even if LLVM frees its handle
    // mid-operation, in-flight calls into the real manager still operate on
    // a live object.
    ObJitAllocator *alloc_ptr = &JITAllocator;
    ObNotifyLoaded *notify_ptr = &NotifyLoaded;
    ObEngineBuilder.setObjectLinkingLayerCreator(
    [alloc_ptr, notify_ptr](ExecutionSession &ES, const Triple &TT) {
      auto ObjLinkingLayer =
          std::make_unique<RTDyldObjectLinkingLayer>(
            ES,
            [alloc_ptr]() -> std::unique_ptr<RuntimeDyld::MemoryManager> {
              // NB: std::unique_ptr usage here is mandated by the LLVM ORC
              // factory signature (returns std::unique_ptr<MemoryManager>).
              // §6.1 exemption applies.
              std::unique_ptr<ObJitMemoryManager> real(new ObJitMemoryManager(*alloc_ptr));
              ObJitMemoryManager *raw = real.get();
              {
                detail::ObJitMemMgrPool &pool = detail::get_mem_mgr_pool();
                std::lock_guard<std::mutex> lk(pool.get_mutex());
                pool.get_mgrs().push_back(std::move(real));
              }
              std::unique_ptr<detail::ObJitMemoryManagerShim> shim(
                  new detail::ObJitMemoryManagerShim(raw));
              return shim;
          });

#if defined(__APPLE__) && defined(__aarch64__)
      // Process all sections (including .eh_frame / __eh_frame) even if they
      // don't contain symbols. Without this, the .eh_frame section is skipped
      // by RuntimeDyld, EH frames are not registered with the system via
      // __register_frame, and PL exception handlers will not work because
      // the unwinder cannot find the personality function or LSDA data for
      // JIT-compiled PL code.
      ObjLinkingLayer->setProcessAllSections(true);
#elif defined(_WIN32)
      ObjLinkingLayer->setProcessAllSections(true);
      ObjLinkingLayer->setAutoClaimResponsibilityForObjectSymbols(true);
      ObjLinkingLayer->setOverrideObjectFlagsWithResponsibilityFlags(true);
#endif
      ObjLinkingLayer->registerJITEventListener(*notify_ptr);
      return ObjLinkingLayer;
    });

    ObEngineBuilder.setCompileFunctionCreator(
      [this] (JITTargetMachineBuilder JTMB)
          -> Expected<std::unique_ptr<IRCompileLayer::IRCompiler>> {
        auto tm = JTMB.createTargetMachine();
        if (!tm) {
          return tm.takeError();
        }
        return std::make_unique<ObPLIRCompiler>(*this, std::move(*tm));
      }
    );

    auto tm_builder_wrapper = JITTargetMachineBuilder::detectHost();

    if (!tm_builder_wrapper) {
      Error err = tm_builder_wrapper.takeError();
      std::string msg = toString(std::move(err));
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("failed to get target machine", K(msg.c_str()));
    } else {
#if defined(__APPLE__) && defined(__aarch64__)
      // Force DwarfCFI exception handling model to ensure .eh_frame and LSDA
      // (Language-Specific Data Area) are generated by the JIT code generator.
      // Without this, on macOS ARM64 the default ExceptionModel is None, which
      // causes compact unwind to be used instead. Compact unwind cannot carry
      // LSDA data, so the PL personality function (ObPLEH::eh_personality)
      // cannot find exception handlers, causing PL DECLARE HANDLER to fail.
      tm_builder_wrapper->getOptions().ExceptionModel = ExceptionHandling::DwarfCFI;
      tm_builder_wrapper->getOptions().MCOptions.EmitDwarfUnwind = EmitDwarfUnwindType::Always;
#endif
      tm_builder_wrapper->setCodeModel(llvm::CodeModel::Large);
      ObEngineBuilder.setJITTargetMachineBuilder(*tm_builder_wrapper);
    }

  return ret;
}

int ObOrcJit::addModule(std::unique_ptr<Module> M, std::unique_ptr<ObLLVMContext> TheContext)
{
  int ret = OB_SUCCESS;

  if (OB_FAIL(create_jit_engine())) {
    LOG_WARN("failed to create jit engine", K(ret));
  } else if (OB_ISNULL(ObJitEngine)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpected NULL jit engine", K(ret), K(lbt()));
  } else {
    Error err = ObJitEngine->addIRModule(ThreadSafeModule{std::move(M), std::move(TheContext)});

    if (err) {
      std::string msg = toString(std::move(err));

      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("failed to add module to jit engine",
               K(ret), K(msg.c_str()));
    }
  }

  return ret;
}

int ObOrcJit::lookup(const std::string &name, ObJITSymbol &symbol)
{
  int ret = OB_SUCCESS;

  if (OB_ISNULL(ObJitEngine)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpected NULL jit engine", K(ret), K(lbt()));
  } else {
    auto value = ObJitEngine->lookup(name);

    if (!value) {
      Error err = value.takeError();

      if (err.isA<SymbolsNotFound>()) {
        ret = OB_ENTRY_NOT_EXIST;
      } else {
        ret = OB_ERR_UNEXPECTED;
      }

      std::string msg = toString(std::move(err));
      LOG_WARN("failed to lookup symbol in jit engine",
        K(ret),
        "name", name.c_str(),
        "msg", msg.c_str());
    } else {
      symbol = JITEvaluatedSymbol(value->getValue(), JITSymbolFlags::Exported);
    }
  }

  return ret;
}

int ObOrcJit::get_function_address(const std::string &name, uint64_t &addr)
{
  int ret = OB_SUCCESS;

  ObJITSymbol sym = nullptr;

  if (OB_FAIL(lookup(name, sym))) {
    LOG_WARN("failed to lookup symbol addr", K(name.c_str()));
  } else {
    auto value = sym.getAddress();

    if (!value) {
      Error err = value.takeError();
      std::string msg = toString(std::move(err));

      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("failed to get symbol address",
               K(ret),
               "name", name.c_str(),
               "msg", msg.c_str());
    } else {
      addr = static_cast<uint64_t>(*value);
    }
  }

  return ret;
}

void ObNotifyLoaded::notifyObjectLoaded(
  ObObjectKey Key,
  const object::ObjectFile &Obj,
  const RuntimeDyld::LoadedObjectInfo &Info)
{
  char *obj_buf = static_cast<char*>(Allocator.alloc(Obj.getData().size()));
  if (OB_NOT_NULL(obj_buf)) {
    MEMCPY(obj_buf, Obj.getData().data(), Obj.getData().size());
    SoObject.assign_ptr(obj_buf, static_cast<ObString::obstr_size_t>(Obj.getData().size()));
  }

  object::OwningBinary<object::ObjectFile> DebugObj = Info.getObjectForDebug(Obj);
  if (DebugObj.getBinary() != nullptr) {
    const char* TmpDebugBuf
      = DebugObj.getBinary()->getMemoryBufferRef().getBufferStart();
    DebugLen
      = DebugObj.getBinary()->getMemoryBufferRef().getBufferSize();
    if (OB_NOT_NULL(
      DebugBuf = static_cast<char*>(Allocator.alloc(DebugLen)))) {
      std::memcpy(DebugBuf, TmpDebugBuf, DebugLen);
    }

    registerDebugInfoToGdb(Key);
  }
}

void ObNotifyLoaded::notifyFreeingObject(ObObjectKey Key)
{
  if (OB_NOT_NULL(DebugBuf)) {
    deregisterDebugInfoFromGdb(Key);
  }
}

int ObNotifyLoaded::initGdbHelper()
{
  int ret = OB_SUCCESS;
  lib::ObMutexGuard guard(AllGdbReg.first);

  if (OB_FAIL(AllGdbReg.second.create(1024, ObMemAttr(OB_SYS_TENANT_ID, "PlGdbHelper")))) {
    LOG_WARN("failed to create AllGdbReg map", K(ret));
  }

  return ret;
}

void ObNotifyLoaded::registerDebugInfoToGdb(ObObjectKey Key)
{
  int ret = OB_SUCCESS;

  lib::ObMutexGuard guard(AllGdbReg.first);
  jit_code_entry buffer = {nullptr, nullptr, DebugBuf, static_cast<uint64_t>(DebugLen)};
  jit_code_entry *entry = nullptr;

  if (OB_FAIL(AllGdbReg.second.set_refactored(Key, buffer))) {
    LOG_WARN("failed to set_refactored to AllGdbReg", K(ret));
  } else if (OB_ISNULL(entry = AllGdbReg.second.get(Key))) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpected NULL entry after set_refactored", K(ret));
  } else {
    jit_code_entry *next = __jit_debug_descriptor.first_entry;
    if (OB_NOT_NULL(next)) {
      next->prev_entry = entry;
    }

    entry->prev_entry = nullptr;
    entry->next_entry = next;

    __jit_debug_descriptor.first_entry = entry;

    __jit_debug_descriptor.relevant_entry = entry;
    __jit_debug_descriptor.action_flag = JIT_REGISTER_FN;
    __jit_debug_register_code();
  }

  LOG_DEBUG("finished registerDebugInfoToGdb", K(ret), K(Key), K(AllGdbReg.second.size()));
}

void ObNotifyLoaded::deregisterDebugInfoFromGdb(ObObjectKey Key)
{
  int ret = OB_SUCCESS;

  lib::ObMutexGuard guard(AllGdbReg.first);

  jit_code_entry *entry = AllGdbReg.second.get(Key);

  if (OB_NOT_NULL(entry)) {
    jit_code_entry *prev = entry->prev_entry;
    jit_code_entry *next = entry->next_entry;

    if (OB_NOT_NULL(prev)) {
      prev->next_entry = next;
    } else {
      __jit_debug_descriptor.first_entry = next;
    }

    if (OB_NOT_NULL(next)) {
      next->prev_entry = prev;
    }

    __jit_debug_descriptor.relevant_entry = entry;
    __jit_debug_descriptor.action_flag = JIT_UNREGISTER_FN;
    __jit_debug_register_code();

    if (OB_FAIL(AllGdbReg.second.erase_refactored(Key))) {
      LOG_WARN("failed to erase jit entry from hashmap", K(ret));
    }
  }
  
  LOG_DEBUG("finished deregisterDebugInfoFromGdb", K(ret), K(Key), K(entry), K(AllGdbReg.second.size()));
}

int ObOrcJit::add_compiled_object(size_t length, const char *ptr) 
{
  int ret = OB_SUCCESS;

  if (OB_FAIL(create_jit_engine())) {
    LOG_WARN("failed to create jit engine", K(ret));
  } else if (OB_ISNULL(ObJitEngine)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpected NULL jit engine", K(ret), K(lbt()));
  } else {
    Error err =ObJitEngine->addObjectFile(
                MemoryBuffer::getMemBuffer(StringRef(ptr, length), "", false));

    if (err) {
      std::string msg = toString(std::move(err));

      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("failed to add compile result to jit engine",
               K(ret), K(msg.c_str()), K(length), K(ptr));
    }
  }

  return ret;
}

int ObOrcJit::set_optimize_level(ObPLOptLevel level)
{
  int ret = OB_SUCCESS;

  if (level <= ObPLOptLevel::INVALID || level > ObPLOptLevel::O3) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpected PLSQL_OPTIMIZE_LEVEL", K(ret), K(level), K(lbt()));
  }

  if (OB_SUCC(ret) && level == ObPLOptLevel::O0) {
    auto &tm_builder = ObEngineBuilder.getJITTargetMachineBuilder();
    if (!tm_builder.has_value()) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("unexpected NULL JITTargetMachineBuilder", K(ret), K(lbt()));
    } else {
      auto &builder = *tm_builder;
#if LLVM_VERSION_MAJOR >= 18
      builder.setCodeGenOptLevel(CodeGenOptLevel::None);
#else
      builder.setCodeGenOptLevel(CodeGenOpt::Level::None);
#endif
      builder.getOptions().EnableFastISel = true;
    }
  }

  return ret;
}

int ObOrcJit::create_jit_engine()
{
  int ret = OB_SUCCESS;

  if (OB_NOT_NULL(ObJitEngine)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpected NOT NULL jit engine", K(ret), K(lbt()));
  } else {
    std::unique_ptr<ObJitGlobalSymbolGenerator> symbol_generator = nullptr;

    auto engine_wrapper = ObEngineBuilder.create();

    if (!engine_wrapper) {
      Error err = engine_wrapper.takeError();
      std::string msg = toString(std::move(err));

      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("failed to build LLVM JIT engine", K(msg.c_str()));
    } else {
      ObJitEngine = std::move(*engine_wrapper);
    }

    if (OB_FAIL(ret)) {
      // do nothing
    } else if (OB_ISNULL(ObJitEngine)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("unexpected NULL jit engine", K(ret));
    } else if (OB_FAIL(ob_jit_make_unique(symbol_generator))) {
      LOG_WARN("failed to make ObJitGlobalSymbolGenerator unique_ptr", K(ret));
    } else {
      ObJitEngine->getMainJITDylib().addGenerator(std::move(symbol_generator));
    }
  }

  return ret;
}

} // namespace core
} // objit
} // oceanbase
