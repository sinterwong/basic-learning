cmake_minimum_required(VERSION 3.15)

function(conan_config_install)
    # Check if Conan is installed
    find_program(CONAN_EXECUTABLE conan)
    if(NOT CONAN_EXECUTABLE)
        message(FATAL_ERROR "Conan not found. Please install Conan first.")
    endif()

    # Check Conan version
    execute_process(
        COMMAND ${CONAN_EXECUTABLE} --version
        OUTPUT_VARIABLE CONAN_VERSION_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE CONAN_VERSION_RESULT
    )
    if(NOT CONAN_VERSION_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to get Conan version.")
    endif()

    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" CONAN_VERSION "${CONAN_VERSION_OUTPUT}")
    if(CONAN_VERSION VERSION_LESS "2.3.0")
        message(FATAL_ERROR "Conan version must be at least 2.3.0. Found: ${CONAN_VERSION}")
    endif()

    # Set up Conan profile if not exists
    execute_process(
        COMMAND ${CONAN_EXECUTABLE} profile path default
        RESULT_VARIABLE CONAN_PROFILE_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(NOT CONAN_PROFILE_RESULT EQUAL 0)
        message(STATUS "Creating default Conan profile...")
        execute_process(
            COMMAND ${CONAN_EXECUTABLE} profile detect --force
            RESULT_VARIABLE CONAN_PROFILE_CREATE_RESULT
        )
        if(NOT CONAN_PROFILE_CREATE_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to create Conan profile.")
        endif()
    endif()

    # Determine the build type
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
    endif()
    message(STATUS "Current build type: ${CMAKE_BUILD_TYPE}")

    # Set Conan user home
    if(DEFINED ENV{CONAN_HOME})
        message(STATUS "CONAN_HOME defined: $ENV{CONAN_HOME}")
    else()
        message(STATUS "CONAN_HOME not defined, using default.")
    endif()

    # Run Conan install
    message(STATUS "Running Conan install...")
    execute_process(
        COMMAND ${CONAN_EXECUTABLE} install ${CMAKE_SOURCE_DIR} 
                -s build_type=Release 
                -s compiler.cppstd=gnu20
                -s:h build_type=${CMAKE_BUILD_TYPE} 
                -b missing
                -c tools.system.package_manager:mode=install
        RESULT_VARIABLE CONAN_INSTALL_RESULT
        OUTPUT_VARIABLE CONAN_INSTALL_OUTPUT
        ERROR_VARIABLE CONAN_INSTALL_ERROR
    )
    if(NOT CONAN_INSTALL_RESULT EQUAL 0)
        message(FATAL_ERROR "Conan install failed.\nError: ${CONAN_INSTALL_ERROR}")
    else()
        message(STATUS "Conan install completed successfully.")
    endif()
endfunction()