if (WIN32 OR APPLE)
    set(SKIP_TOOLCHAIN TRUE)
endif()

if (SKIP_TOOLCHAIN)
    # skip toolchain so that code insight can work properly
    cmake_minimum_required(VERSION 3.0)
    project(graphvite LANGUAGES CXX)
    include_directories(include)
    add_subdirectory(src)
    return()
endif()

cmake_minimum_required(VERSION 3.12)
project(graphvite LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(FindCUDA)

find_package(Glog)
find_package(GFlags)
find_package(PythonLibsNew REQUIRED)
find_package(pybind11 REQUIRED)

# CUDA_ROOT & CUDA_ARCH
get_filename_component(CUDA_ROOT ${CMAKE_CUDA_COMPILER} DIRECTORY)
get_filename_component(CUDA_ROOT ${CUDA_ROOT} DIRECTORY)
if (NOT ALL_ARCH)
    cuda_select_nvcc_arch_flags(CUDA_ARCH Auto)
else()
    cuda_select_nvcc_arch_flags(CUDA_ARCH 3.5 5.0 6.0 7.0)
endif()
string(REPLACE ";" " " CUDA_ARCH "${CUDA_ARCH}")

# add externals
include(ExternalProject)
set(EXTERNAL_DIR ${PROJECT_SOURCE_DIR}/external)

if (NOT UPDATE)
    set(SKIP_UPDATE TRUE)
else()
    set(SKIP_UPDATE FALSE)
endif()

# glog
if (NOT GLOG_FOUND)
    message(WARNING "Can't find glog library. It will be installed from git repository.")
    ExternalProject_Add(glog
            GIT_REPOSITORY      https://github.com/google/glog.git
            UPDATE_DISCONNECTED ${SKIP_UPDATE}
            PREFIX              glog
            SOURCE_DIR          ${EXTERNAL_DIR}/glog
            BINARY_DIR          ${PROJECT_BINARY_DIR}/glog
            INSTALL_DIR         ${PROJECT_BINARY_DIR}/glog
            CONFIGURE_COMMAND   test -e Makefile && exit ||
                                cd ${EXTERNAL_DIR}/glog && ./autogen.sh && cd - &&
                                ${EXTERNAL_DIR}/glog/configure
            BUILD_COMMAND       test -e .libs/libglog.a || make -j
            INSTALL_COMMAND     test -e include/glog/logging.h || make install prefix=${PROJECT_BINARY_DIR}/glog)
    include_directories(${PROJECT_BINARY_DIR}/glog/include)
    link_directories(${PROJECT_BINARY_DIR}/glog/.libs)
else()
    get_filename_component(GLOG_LIBRARY_DIR ${GLOG_LIBRARY} DIRECTORY)
    include_directories(${GLOG_INCLUDE_DIR})
    link_directories(${GLOG_LIBRARY_DIR})
endif ()

# gflags
if (NOT GFLAGS_FOUND)
    message(WARNING "Can't find gflags library. It will be installed from git repository.")
    ExternalProject_Add(gflags
            GIT_REPOSITORY      https://github.com/gflags/gflags
            UPDATE_DISCONNECTED ${SKIP_UPDATE}
            PREFIX              gflags
            SOURCE_DIR          ${EXTERNAL_DIR}/gflags
            BINARY_DIR          ${PROJECT_BINARY_DIR}/gflags
            INSTALL_DIR         ${PROJECT_BINARY_DIR}/gflags
            CONFIGURE_COMMAND   test -e Makefile ||
                                cmake ${EXTERNAL_DIR}/gflags -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}/gflags
            BUILD_COMMAND       test -e lib/libgflags.a || make -j
            INSTALL_COMMAND     test -e include/gflags/gflags.h || make install)
    include_directories(${PROJECT_BINARY_DIR}/gflags/include)
    link_directories(${PROJECT_BINARY_DIR}/gflags/lib)
else()
    get_filename_component(GFLAGS_LIBRARY_DIR ${GFLAGS_LIBRARY} DIRECTORY)
    include_directories(${GFLAGS_INCLUDE_DIR})
    link_directories(${GFLAGS_LIBRARY_DIR})
endif()

# faiss
if (NOT FAISS_PATH)
    if (NOT NO_FAISS)
        ExternalProject_Add(faiss
                GIT_REPOSITORY      https://github.com/facebookresearch/faiss.git
                UPDATE_DISCONNECTED ${SKIP_UPDATE}
                PREFIX              faiss
                SOURCE_DIR          ${EXTERNAL_DIR}/faiss
                BINARY_DIR          ${EXTERNAL_DIR}/faiss
                INSTALL_DIR         ${PROJECT_BINARY_DIR}/faiss
                CONFIGURE_COMMAND   test -e makefile.inc ||
                                    ${EXTERNAL_DIR}/faiss/configure --with-cuda=${CUDA_ROOT} --with-cuda-arch=${CUDA_ARCH}
                BUILD_COMMAND       test -e libfaiss.so || make -j
                INSTALL_COMMAND     test -e ${PROJECT_BINARY_DIR}/faiss/include/faiss/gpu/GpuIndexFlat.h ||
                                    make install prefix=${PROJECT_BINARY_DIR}/faiss)
        include_directories(${PROJECT_BINARY_DIR}/faiss/include)
        link_directories(${PROJECT_BINARY_DIR}/faiss/lib)
        set(FAISS_LIBRARY ${PROJECT_BINARY_DIR}/faiss/lib/libfaiss.so)
    endif()
else()
    get_filename_component(FAISS_PARENT ${FAISS_PATH} DIRECTORY)
    include_directories(${FAISS_PARENT})
    link_directories(${FAISS_PATH})
    set(FAISS_LIBRARY ${FAISS_PATH}/libfaiss.so)
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# compilation flags
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Og")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -D NDEBUG")

set(CMAKE_CUDA_FLAGS "-std=c++11")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH}")

foreach (MACRO FAST_COMPILE NO_FAISS)
    if (${${MACRO}})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D${MACRO}")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D${MACRO}")
    endif()
endforeach()

# output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
# symbolic link to faiss
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
execute_process(COMMAND ln -sf ${FAISS_LIBRARY} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libfaiss.so)

# project directories
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(include)
add_subdirectory(src)