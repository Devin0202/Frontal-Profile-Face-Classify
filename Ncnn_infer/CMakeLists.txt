cmake_minimum_required(VERSION 3.10)
project(NcnnInfer)

set(BASICLIBS opencv_highgui opencv_imgcodecs opencv_imgproc opencv_core)

IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    MESSAGE(STATUS "current platform: Linux ")

    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g")

    set(MAIN ./test.cpp  ./quality_assessment.cc)

    set(3RDLIBS libwebp IlmImf libjpeg-turbo libpng libjasper libtiff tbb z)

    set(NCNNLIBS ncnn)

    FIND_PACKAGE(OpenMP REQUIRED)
    if(OPENMP_FOUND)
        message("OPENMP FOUND")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    else()
        message("OPENMP NOT FOUND")
    endif()

    include_directories(/home/devin/MyGit/ncnn-linux-install/include/ncnn)

    link_directories(/home/devin/MyGit/ncnn-linux-install/lib)

    include_directories(/home/devin/MyGit/opencv346-a-linux-install/include)
    
    link_directories(/home/devin/MyGit/opencv346-a-linux-install/lib)

    link_directories(/home/devin/MyGit/opencv346-a-linux-install/3rdparty/lib)

    set(BASICLIBS ${BASICLIBS} ${3RDLIBS} dl)
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Windows")
	MESSAGE(STATUS "current platform: Windows")
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Android")
    MESSAGE(STATUS "current platform: Android")

    FIND_PACKAGE(OpenMP REQUIRED)
    if(OPENMP_FOUND)
        message("OPENMP FOUND")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    else()
        message("OPENMP NOT FOUND")
    endif()

    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -s")

    set(MAIN ./main-android.cpp)

    set(NCNNLIBS ncnn)

    set(3RDLIBS libwebp IlmImf libjpeg-turbo libpng libjasper libtiff tegra_hal z log)

    include_directories(/home/devin/MyGit/opencv-3.4.6/build_android_v8/install/sdk/native/jni/include)

    link_directories(/home/devin/MyGit/opencv-3.4.6/build_android_v8/install/sdk/native/staticlibs/arm64-v8a)

    link_directories(/home/devin/MyGit/opencv-3.4.6/build_android_v8/install/sdk/native/3rdparty/libs/arm64-v8a)

    include_directories(/home/devin/MyGit/ncnn-android-install/include/ncnn)

    link_directories(/home/devin/MyGit/ncnn-android-install/lib)

    set(BASICLIBS ${BASICLIBS} ${3RDLIBS})
ELSE ()
    MESSAGE(STATUS "current platform: None")
ENDIF ()

add_executable(main ${MAIN})
target_link_libraries(main  ${BASICLIBS} ${NCNNLIBS})