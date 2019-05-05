wget https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage
wget https://github.com/linuxdeploy/linuxdeploy-plugin-appimage/releases/download/continuous/linuxdeploy-plugin-appimage-x86_64.AppImage

chmod +x linuxdeploy-*.AppImage
mkdir -p ci/appimage/appdir/usr/bin
cp build_onnx2bnn/tools/onnx2bnn/onnx2bnn ci/appimage/appdir/usr/bin/
./linuxdeploy-x86_64.AppImage --appdir ci/appimage/appdir -d ci/appimage/onnx2bnn.desktop -i ci/appimage/onnx2bnn.png --output appimage
mv `ls onnx2bnn-*.AppImage` onnx2bnn.AppImage
