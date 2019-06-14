// Copyright 2019 JD.com Inc. JD AI

#include "Pad.h"

#include <dabnn/net.h>
#include <dabnn/pad.h>

namespace bnn {
void Pad::forward_impl() const { pad(*input_mat, pad_h, pad_w, *output_mat); }

}  // namespace bnn
