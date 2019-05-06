// Copyright 2019 JD.com Inc. JD AI

package me.daquexian.dabnn;

import android.content.res.AssetManager;

public class Net {
    static {
        System.loadLibrary("dabnn_jni");
    }
    public Net() {
        initHandle();
    }
    @Override
    protected void finalize() throws Throwable {
        dispose();
        super.finalize();
    }
    private long nativeHandle;
    private native void initHandle();
    public native Net readFile(String filename);
    public native Net readAsset(AssetManager assetManager, String filename);
    public native float[] getBlob(String blobName);
    public native void predict(float[] input);
    public native void dispose();
}
