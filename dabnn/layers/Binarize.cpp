// Copyright 2019 JD.com Inc. JD AI

#include "Binarize.h"

#include <dabnn/bitpack.h>
#include <dabnn/net.h>

namespace bnn {
void Binarize::forward_impl() const { 
    if (net_.lock()->new_bitpack) {
        ::pack_mat(*input_mat, *output_mat); 
    } else {
        ::pack_mat_64(*input_mat, *output_mat); 
    }
}

}  // namespace bnn
