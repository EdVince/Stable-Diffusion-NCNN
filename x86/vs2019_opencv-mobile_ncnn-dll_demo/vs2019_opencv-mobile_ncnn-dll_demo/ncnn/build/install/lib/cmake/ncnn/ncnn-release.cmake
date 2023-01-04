#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ncnn" for configuration "Release"
set_property(TARGET ncnn APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ncnn PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ncnn.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ncnn.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS ncnn )
list(APPEND _IMPORT_CHECK_FILES_FOR_ncnn "${_IMPORT_PREFIX}/lib/ncnn.lib" "${_IMPORT_PREFIX}/bin/ncnn.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
