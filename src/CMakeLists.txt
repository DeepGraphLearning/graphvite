if (WIN32)
    add_library(graphvite graphvite.cu)
else ()
    add_library(graphvite SHARED graphvite.cu)
    set_target_properties(graphvite PROPERTIES
            CXX_VISIBILITY_PRESET   "hidden"
            CUDA_VISIBILITY_PRESET  "hidden"
            LINK_FLAGS              "-flto -Wl,-rpath=$ORIGIN"
            OUTPUT_NAME             graphvite)

    target_link_libraries(graphvite pthread curand glog.so)
    if (FAISS_LIBRARY)
        target_link_libraries(graphvite faiss.so)
    endif()
    target_compile_options(graphvite PRIVATE "-Xcompiler=-fno-fat-lto-objects") # -flto
endif ()