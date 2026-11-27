// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

//! Zero-dependency dynamic CUDA Driver API wrapper (`libcuda.so.1`) via
//! `libc::dlopen`.
//!
//! Compiles embedded PTX directly on the installed NVIDIA driver (no `nvcc`,
//! no CUDA Toolkit, no `cmake`, no license server) and returns `None` cleanly
//! when no CUDA driver or GPU is present.

use core::ffi::{c_char, c_int, c_uint, c_void};
use core::ptr;
use std::ffi::CString;

pub(crate) type CuResult = c_int;
pub(crate) type CuDevice = c_int;
pub(crate) type CuContext = *mut c_void;
pub(crate) type CuModule = *mut c_void;
pub(crate) type CuFunction = *mut c_void;
pub(crate) type CuDevicePtr = u64;

pub(crate) const CUDA_SUCCESS: CuResult = 0;

// CUjit_option enum values from cuda.h
const CU_JIT_ERROR_LOG_BUFFER: c_int = 5;
const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_int = 6;
// CUlimit enum value from cuda.h
const CU_LIMIT_STACK_SIZE: c_int = 0;

type FnCuInit = unsafe extern "C" fn(c_uint) -> CuResult;
type FnCuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CuResult;
type FnCuDeviceGet = unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult;
type FnCuDeviceGetName =
    unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult;
type FnCuDevicePrimaryCtxRetain =
    unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult;
type FnCuCtxSetCurrent = unsafe extern "C" fn(CuContext) -> CuResult;
type FnCuCtxSetLimit = unsafe extern "C" fn(c_int, usize) -> CuResult;
type FnCuCtxSynchronize = unsafe extern "C" fn() -> CuResult;
type FnCuModuleLoadDataEx = unsafe extern "C" fn(
    *mut CuModule,
    *const c_void,
    c_uint,
    *mut c_int,
    *mut *mut c_void,
) -> CuResult;
type FnCuModuleGetFunction =
    unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
type FnCuMemAlloc = unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult;
type FnCuMemFree = unsafe extern "C" fn(CuDevicePtr) -> CuResult;
type FnCuMemAllocHost =
    unsafe extern "C" fn(*mut *mut c_void, usize) -> CuResult;
type FnCuMemFreeHost = unsafe extern "C" fn(*mut c_void) -> CuResult;
type FnCuMemcpyHtoD =
    unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult;
type FnCuMemcpyDtoH =
    unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult;
type FnCuMemsetD8 = unsafe extern "C" fn(CuDevicePtr, u8, usize) -> CuResult;
type FnCuLaunchKernel = unsafe extern "C" fn(
    CuFunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CuResult;

pub(crate) struct CudaDriver {
    ctx: CuContext,
    #[allow(dead_code)]
    pub(crate) device_name: String,
    ctx_set_current: FnCuCtxSetCurrent,
    #[cfg_attr(not(test), allow(dead_code))]
    ctx_synchronize: FnCuCtxSynchronize,
    module_load_data_ex: FnCuModuleLoadDataEx,
    module_get_function: FnCuModuleGetFunction,
    mem_alloc: FnCuMemAlloc,
    mem_free: FnCuMemFree,
    mem_alloc_host: Option<FnCuMemAllocHost>,
    mem_free_host: Option<FnCuMemFreeHost>,
    memcpy_htod: FnCuMemcpyHtoD,
    memcpy_dtoh: FnCuMemcpyDtoH,
    memset_d8: FnCuMemsetD8,
    launch_kernel: FnCuLaunchKernel,
}

// SAFETY: CUDA primary contexts are thread-safe once bound via
// `cuCtxSetCurrent` on the calling host thread.
unsafe impl Send for CudaDriver {}
unsafe impl Sync for CudaDriver {}

#[cfg(unix)]
unsafe fn dlsym_fn<T: Copy>(handle: *mut c_void, name: &str) -> Option<T> {
    let cname = CString::new(name).ok()?;
    // SAFETY: valid library handle and NUL-terminated symbol name.
    let sym = unsafe { libc::dlsym(handle, cname.as_ptr()) };
    if sym.is_null() {
        return None;
    }
    // SAFETY: caller matches `T` to the exported NVIDIA Driver API signature.
    Some(unsafe { core::mem::transmute_copy::<*mut c_void, T>(&sym) })
}

impl CudaDriver {
    #[cfg(unix)]
    pub(crate) fn load() -> Option<Self> {
        let candidates = [
            "libcuda.so.1",
            "/usr/lib/wsl/lib/libcuda.so.1",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "libcuda.so",
        ];
        let mut handle = ptr::null_mut();
        for name in candidates {
            let cname = CString::new(name).ok()?;
            // SAFETY: NUL-terminated library path.
            handle = unsafe { libc::dlopen(cname.as_ptr(), libc::RTLD_NOW) };
            if !handle.is_null() {
                break;
            }
        }
        if handle.is_null() {
            return None;
        }

        // SAFETY: symbol names and signatures match CUDA Driver API (`cuda.h`).
        unsafe {
            let cu_init: FnCuInit = dlsym_fn(handle, "cuInit")?;
            let cu_device_get_count: FnCuDeviceGetCount =
                dlsym_fn(handle, "cuDeviceGetCount")?;
            let cu_device_get: FnCuDeviceGet = dlsym_fn(handle, "cuDeviceGet")?;
            let cu_device_get_name: FnCuDeviceGetName =
                dlsym_fn(handle, "cuDeviceGetName")?;
            let cu_primary_ctx_retain: FnCuDevicePrimaryCtxRetain =
                dlsym_fn(handle, "cuDevicePrimaryCtxRetain")?;
            let ctx_set_current: FnCuCtxSetCurrent =
                dlsym_fn(handle, "cuCtxSetCurrent")?;
            let ctx_set_limit: Option<FnCuCtxSetLimit> =
                dlsym_fn(handle, "cuCtxSetLimit");
            let ctx_synchronize: FnCuCtxSynchronize =
                dlsym_fn(handle, "cuCtxSynchronize")?;
            let module_load_data_ex: FnCuModuleLoadDataEx =
                dlsym_fn(handle, "cuModuleLoadDataEx")?;
            let module_get_function: FnCuModuleGetFunction =
                dlsym_fn(handle, "cuModuleGetFunction")?;
            let mem_alloc: FnCuMemAlloc = dlsym_fn(handle, "cuMemAlloc_v2")?;
            let mem_free: FnCuMemFree = dlsym_fn(handle, "cuMemFree_v2")?;
            let mem_alloc_host: Option<FnCuMemAllocHost> =
                dlsym_fn(handle, "cuMemAllocHost_v2");
            let mem_free_host: Option<FnCuMemFreeHost> =
                dlsym_fn(handle, "cuMemFreeHost");
            let memcpy_htod: FnCuMemcpyHtoD =
                dlsym_fn(handle, "cuMemcpyHtoD_v2")?;
            let memcpy_dtoh: FnCuMemcpyDtoH =
                dlsym_fn(handle, "cuMemcpyDtoH_v2")?;
            let memset_d8: FnCuMemsetD8 = dlsym_fn(handle, "cuMemsetD8_v2")?;
            let launch_kernel: FnCuLaunchKernel =
                dlsym_fn(handle, "cuLaunchKernel")?;

            if cu_init(0) != CUDA_SUCCESS {
                return None;
            }
            let mut count = 0;
            if cu_device_get_count(&mut count) != CUDA_SUCCESS || count <= 0 {
                return None;
            }
            let ordinal: c_int = std::env::var("PLONK_CUDA_DEVICE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            if ordinal < 0 || ordinal >= count {
                return None;
            }
            let mut dev = 0;
            if cu_device_get(&mut dev, ordinal) != CUDA_SUCCESS {
                return None;
            }
            let mut name_buf = [0u8; 128];
            let device_name = if cu_device_get_name(
                name_buf.as_mut_ptr().cast::<c_char>(),
                name_buf.len() as c_int,
                dev,
            ) == CUDA_SUCCESS
            {
                let nul = name_buf
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(name_buf.len());
                String::from_utf8_lossy(&name_buf[..nul]).into_owned()
            } else {
                "CUDA GPU".to_string()
            };

            let mut ctx = ptr::null_mut();
            if cu_primary_ctx_retain(&mut ctx, dev) != CUDA_SUCCESS
                || ctx.is_null()
            {
                return None;
            }
            if ctx_set_current(ctx) != CUDA_SUCCESS {
                return None;
            }
            if let Some(set_limit) = ctx_set_limit {
                let _ = set_limit(CU_LIMIT_STACK_SIZE, 16384);
            }

            Some(Self {
                ctx,
                device_name,
                ctx_set_current,
                ctx_synchronize,
                module_load_data_ex,
                module_get_function,
                mem_alloc,
                mem_free,
                mem_alloc_host,
                mem_free_host,
                memcpy_htod,
                memcpy_dtoh,
                memset_d8,
                launch_kernel,
            })
        }
    }

    #[cfg(not(unix))]
    pub(crate) fn load() -> Option<Self> {
        None
    }

    #[inline]
    pub(crate) fn bind_ctx(&self) -> Result<(), CuResult> {
        // SAFETY: `self.ctx` is a retained primary CUDA context.
        let rc = unsafe { (self.ctx_set_current)(self.ctx) };
        if rc == CUDA_SUCCESS { Ok(()) } else { Err(rc) }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn synchronize(&self) -> Result<(), CuResult> {
        // SAFETY: context is bound on calling thread.
        let rc = unsafe { (self.ctx_synchronize)() };
        if rc == CUDA_SUCCESS { Ok(()) } else { Err(rc) }
    }

    pub(crate) fn load_ptx(&self, ptx: &str) -> Result<CuModule, String> {
        self.bind_ctx()
            .map_err(|e| format!("cuCtxSetCurrent: {e}"))?;
        let cptx =
            CString::new(ptx).map_err(|_| "PTX contains NUL".to_string())?;
        let mut err_buf = vec![0u8; 8192];
        let mut err_size = err_buf.len();
        let mut options =
            [CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES];
        let mut values: [*mut c_void; 2] = [
            err_buf.as_mut_ptr().cast::<c_void>(),
            err_size as *mut c_void,
        ];
        let mut module = ptr::null_mut();
        // SAFETY: valid PTX C-string and JIT option arrays.
        let rc = unsafe {
            (self.module_load_data_ex)(
                &mut module,
                cptx.as_ptr().cast::<c_void>(),
                options.len() as c_uint,
                options.as_mut_ptr(),
                values.as_mut_ptr(),
            )
        };
        if rc != CUDA_SUCCESS || module.is_null() {
            err_size = (values[1] as usize).min(err_buf.len());
            let nul = err_buf[..err_size]
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(err_size);
            let log = String::from_utf8_lossy(&err_buf[..nul]);
            return Err(format!("cuModuleLoadDataEx failed (rc={rc}): {log}"));
        }
        Ok(module)
    }

    pub(crate) fn get_function(
        &self,
        module: CuModule,
        name: &str,
    ) -> Result<CuFunction, CuResult> {
        let cname = CString::new(name).map_err(|_| -1)?;
        let mut func = ptr::null_mut();
        // SAFETY: valid module and NUL-terminated kernel name.
        let rc = unsafe {
            (self.module_get_function)(&mut func, module, cname.as_ptr())
        };
        if rc == CUDA_SUCCESS && !func.is_null() {
            Ok(func)
        } else {
            Err(rc)
        }
    }

    pub(crate) fn alloc_bytes(
        &self,
        bytes: usize,
    ) -> Result<CuDevicePtr, CuResult> {
        let mut ptr: CuDevicePtr = 0;
        let alloc_bytes = bytes.max(16);
        // SAFETY: context bound; allocates device memory.
        let rc = unsafe { (self.mem_alloc)(&mut ptr, alloc_bytes) };
        if rc == CUDA_SUCCESS && ptr != 0 {
            Ok(ptr)
        } else {
            Err(rc)
        }
    }

    pub(crate) fn free(&self, ptr: CuDevicePtr) {
        if ptr != 0 {
            let _ = self.bind_ctx();
            // SAFETY: frees pointer allocated by `cuMemAlloc_v2`.
            unsafe {
                let _ = (self.mem_free)(ptr);
            }
        }
    }

    pub(crate) fn alloc_host_pinned(
        &self,
        bytes: usize,
    ) -> Option<*mut c_void> {
        let alloc = self.mem_alloc_host?;
        let mut ptr: *mut c_void = ptr::null_mut();
        // SAFETY: context bound; allocates page-locked host memory.
        let rc = unsafe { alloc(&mut ptr, bytes.max(64)) };
        if rc == CUDA_SUCCESS && !ptr.is_null() {
            Some(ptr)
        } else {
            None
        }
    }

    pub(crate) fn free_host_pinned(&self, ptr: *mut c_void) {
        if !ptr.is_null()
            && let Some(free) = self.mem_free_host
        {
            let _ = self.bind_ctx();
            // SAFETY: frees pointer allocated by `cuMemAllocHost_v2`.
            unsafe {
                let _ = free(ptr);
            }
        }
    }

    pub(crate) fn copy_htod<T>(
        &self,
        dst: CuDevicePtr,
        src: &[T],
    ) -> Result<(), CuResult> {
        let bytes = core::mem::size_of_val(src);
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: caller ensures `dst` has at least `bytes` allocated.
        let rc = unsafe {
            (self.memcpy_htod)(dst, src.as_ptr().cast::<c_void>(), bytes)
        };
        if rc == CUDA_SUCCESS { Ok(()) } else { Err(rc) }
    }

    pub(crate) fn copy_dtoh<T>(
        &self,
        dst: &mut [T],
        src: CuDevicePtr,
    ) -> Result<(), CuResult> {
        let bytes = core::mem::size_of_val(dst);
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: caller ensures `src` has at least `bytes` allocated.
        let rc = unsafe {
            (self.memcpy_dtoh)(dst.as_mut_ptr().cast::<c_void>(), src, bytes)
        };
        if rc == CUDA_SUCCESS { Ok(()) } else { Err(rc) }
    }

    pub(crate) fn memset_zero(
        &self,
        dst: CuDevicePtr,
        bytes: usize,
    ) -> Result<(), CuResult> {
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: caller ensures `dst` has at least `bytes` allocated.
        let rc = unsafe { (self.memset_d8)(dst, 0, bytes) };
        if rc == CUDA_SUCCESS { Ok(()) } else { Err(rc) }
    }

    /// Launch a 1D/2D/3D CUDA kernel on the default stream.
    ///
    /// # Safety
    /// `params` must match the kernel's PTX parameter signature exactly.
    pub(crate) unsafe fn launch(
        &self,
        func: CuFunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<(), CuResult> {
        // SAFETY: forwarded to `cuLaunchKernel` with caller-validated params.
        let rc = unsafe {
            (self.launch_kernel)(
                func,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_mem_bytes,
                ptr::null_mut(),
                params.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        if rc == CUDA_SUCCESS { Ok(()) } else { Err(rc) }
    }
}

/// RAII device buffer owned by a `CudaDriver`.
pub(crate) struct CudaBuf {
    drv: &'static CudaDriver,
    pub(crate) ptr: CuDevicePtr,
    pub(crate) bytes: usize,
}

// SAFETY: CUDA device pointers are immutable addresses synchronized via the
// driver context.
unsafe impl Send for CudaBuf {}
unsafe impl Sync for CudaBuf {}

impl CudaBuf {
    pub(crate) fn alloc(
        drv: &'static CudaDriver,
        bytes: usize,
    ) -> Result<Self, CuResult> {
        let ptr = drv.alloc_bytes(bytes)?;
        Ok(Self { drv, ptr, bytes })
    }

    pub(crate) fn ensure_capacity(
        &mut self,
        bytes: usize,
    ) -> Result<(), CuResult> {
        if self.bytes >= bytes {
            return Ok(());
        }
        let new_ptr = self.drv.alloc_bytes(bytes)?;
        self.drv.free(self.ptr);
        self.ptr = new_ptr;
        self.bytes = bytes;
        Ok(())
    }
}

impl Drop for CudaBuf {
    fn drop(&mut self) {
        self.drv.free(self.ptr);
    }
}

/// Reusable page-locked (pinned) host buffer for full-speed PCIe DMA transfers
/// with automatic fallback to standard heap memory if `cuMemAllocHost_v2` is
/// unavailable.
pub(crate) struct CudaHostBuf<T: Copy + Default> {
    drv: &'static CudaDriver,
    pinned_ptr: *mut T,
    pinned_len: usize,
    fallback: Vec<T>,
}

// SAFETY: `CudaHostBuf` uniquely owns its allocation and is only accessed
// under a `Mutex`.
unsafe impl<T: Copy + Default + Send> Send for CudaHostBuf<T> {}

impl<T: Copy + Default> CudaHostBuf<T> {
    pub(crate) fn new(drv: &'static CudaDriver) -> Self {
        Self {
            drv,
            pinned_ptr: ptr::null_mut(),
            pinned_len: 0,
            fallback: Vec::new(),
        }
    }

    pub(crate) fn ensure_len(&mut self, len: usize) -> &mut [T] {
        if len == 0 {
            return &mut [];
        }
        if !self.pinned_ptr.is_null() && self.pinned_len >= len {
            // SAFETY: `pinned_ptr` is aligned and valid for `pinned_len` `T`s.
            return unsafe {
                core::slice::from_raw_parts_mut(self.pinned_ptr, len)
            };
        }
        let bytes = len.saturating_mul(core::mem::size_of::<T>());
        if let Some(raw) = self.drv.alloc_host_pinned(bytes) {
            if !self.pinned_ptr.is_null() {
                self.drv.free_host_pinned(self.pinned_ptr.cast());
            }
            self.pinned_ptr = raw.cast::<T>();
            self.pinned_len = len;
            self.fallback.clear();
            // SAFETY: `cuMemAllocHost_v2` returns 4KB-aligned host memory.
            return unsafe {
                core::slice::from_raw_parts_mut(self.pinned_ptr, len)
            };
        }
        if self.fallback.len() < len {
            self.fallback.resize(len, T::default());
        }
        &mut self.fallback[..len]
    }
}

impl<T: Copy + Default> Drop for CudaHostBuf<T> {
    fn drop(&mut self) {
        if !self.pinned_ptr.is_null() {
            self.drv.free_host_pinned(self.pinned_ptr.cast());
        }
    }
}
