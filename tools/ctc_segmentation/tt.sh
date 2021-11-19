NEMO_NORMALIZATION="file.nemo"
    if [[ ${NEMO_NORMALIZATION,,} == *".nemo" ]]; then
      echo "NEMO";
    else
      echo "NOT NEMO";
    fi
