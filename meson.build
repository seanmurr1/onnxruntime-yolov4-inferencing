project('gst-ort-plugin', 'cpp')

onnxrt_dep = dependency('libonnxruntime')

if onnxrt_dep.found()
	onnxrt_include_root = onnxrt_dep.get_pkgconfig_variable('includedir')
	onnxrt_includes = [onnxrt_include_root / 'core/session', onnxrt_include_root / 'core']
	onnxrt_dep_args = []
	
	compiler = meson.get_compiler('cpp')
	if compiler.has_header(onnxrt_include_root / 'core/providers/cuda/cuda_provider_factory.h')
	  onnxrt_dep_args = ['-DGST_ML_ONNX_RUNTIME_HAVE_CUDA']
	endif

  opencvdep = dependency('opencv4')

    executable('driver', 
    'src/driver.cpp',
    'src/yolov4.cpp',
    c_args : [],
    cpp_args: onnxrt_dep_args,
    link_args : [],
    include_directories : [onnxrt_includes, 'include'],
    dependencies : [onnxrt_dep, opencvdep])

 endif

