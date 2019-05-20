if [[ "$OSTYPE" == "drawin*" ]]; then
    echo $(sysctl -n hw.physicalcpu)
else
    echo $(nproc)
fi
