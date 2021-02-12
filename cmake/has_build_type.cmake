function(has_build_type BT)
  set(
    AZEBAN_HAS_DEFINED_${BT}
    (DEFINED CMAKE_CXX_FLAGS_${BT} AND DEFINDED CMAKE_EXE_LINKER_FLAGS_${BT})
    PARRENT_SCOPE
  )
endfunction()
