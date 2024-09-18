cmake_minimum_required(VERSION 3.16.3)
project(HalideFilters)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

option(Module_HalideFilters_USE_AUTOSCHEDULER "Use auto-schedulers for Halide filters" OFF)
option(Module_HalideFilters_TEST_GPU "Run GPU tests" OFF)

# Update the following variables to update the version of Halide used
set(HALIDE_VERSION "18.0.0")
set(HALIDE_VERSION_COMMIT "8c651b459a4e3744b413c23a29b5c5d968702bb7")
# import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
set(HASH_linux_x86_64 "8e491e2f9ac7c138482bb344cb291c708632718dfee1dd7f9c3749b3607e41c7")
set(HASH_linux_x86_32 "d3594550b4fccdcebf3cadf42558b816c6909cd1ad4978c58f90d74dd56cf372")
set(HASH_linux_arm_64 "89083e890f47cd9cef983d5555111af4c9628e89b1ce24712dc090edb89c3dd8")
set(HASH_linux_arm_32 "f01a3d8c861c589fcba58e000f870f2852a1f1ce293e9f43746136d1b0efc7ac")
set(HASH_osx_x86_64 "708a2fadab4ba0556c4c82ea63c5bc030d3c8f9d223b44c033d0ed1b1109a55b")
set(HASH_osx_arm_64 "d96831794ba91455a19d1d5832f175c6913d7238cb70dadfdc0e1475ccd9e1e6")
set(HASH_windows_x86_64 "23994ec62902b1558c97201302bfcbba0968ba474344df4668674ae89e309b35")
set(HASH_windows_x86_32 "297b6d1ec8b35b0e99e499bc0210e65b1b1d488662ea5007cd707e4627afceda")

# Set intruction set
string(FIND "${CMAKE_SYSTEM_PROCESSOR}" "x86" IS_X86)
string(FIND "${CMAKE_SYSTEM_PROCESSOR}" "AMD" IS_AMD)
string(FIND "${CMAKE_SYSTEM_PROCESSOR}" "arm" IS_ARM)
if (NOT IS_X86 EQUAL -1 OR NOT IS_AMD EQUAL -1)
  set(INSTRUCTION_SET "x86")
elseif (NOT IS_ARM EQUAL -1)
  set(INSTRUCTION_SET "arm")
else()
  message(FATAL_ERROR "Cannot infer instruction set for Halide release")
endif ()

# Check the bit version
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(BIT_VERSION "64")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(BIT_VERSION "32")
else()
  message(FATAL_ERROR "Cannot infer bitness for Halide release")
endif()

# set operating system
string(TOLOWER "${CMAKE_SYSTEM_NAME}" OS)
if (${OS} STREQUAL "darwin")
  set(OS "osx")
endif ()

# set file extension
set(FILE_EXTENSION "tar.gz")
if (WIN32)
  set(FILE_EXTENSION "zip")
endif ()

set(HALIDE_URL "https://github.com/halide/Halide/releases/download/v${HALIDE_VERSION}/Halide-${HALIDE_VERSION}-${INSTRUCTION_SET}-${BIT_VERSION}-${OS}-${HALIDE_VERSION_COMMIT}.${FILE_EXTENSION}")
set(HALIDE_URL_HASH "${HASH_${OS}_${INSTRUCTION_SET}_${BIT_VERSION}}")

include(FetchContent)
FetchContent_Declare(
  halide
  URL ${HALIDE_URL}
  URL_HASH SHA256=${HALIDE_URL_HASH}
  DOWNLOAD_EXTRACT_TIMESTAMP YES
)
FetchContent_MakeAvailable(halide)

set(Halide_DIR "${halide_SOURCE_DIR}/lib/cmake/Halide")
set(HalideHelpers_DIR "${halide_SOURCE_DIR}/lib/cmake/HalideHelpers")

set(HalideFilters_LIBRARIES HalideFilters)

if(NOT ITK_SOURCE_DIR)
  find_package(ITK REQUIRED)
  list(APPEND CMAKE_MODULE_PATH ${ITK_CMAKE_DIR})
  include(ITKModuleExternal)
else()
  set(ITK_DIR ${CMAKE_BINARY_DIR})
  itk_module_impl()
endif()

itk_module_examples()
