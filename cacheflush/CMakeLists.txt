add_library(cacheflush cacheflush.c)

target_include_directories(cacheflush PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )
