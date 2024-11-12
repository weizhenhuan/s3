#!/bin/bash
echo "[Ascend310B1] Generating IsClose_c727970057bb5b48903c1f9612cfc198 ..."
opc $1 --main_func=is_close --input_param=/home/HwHiAiUser/s3/IsClose/build_out/op_kernel/binary/ascend310b/gen/IsClose_c727970057bb5b48903c1f9612cfc198_param.json --soc_version=Ascend310B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/IsClose_c727970057bb5b48903c1f9612cfc198.json ; then
  echo "$2/IsClose_c727970057bb5b48903c1f9612cfc198.json not generated!"
  exit 1
fi

if ! test -f $2/IsClose_c727970057bb5b48903c1f9612cfc198.o ; then
  echo "$2/IsClose_c727970057bb5b48903c1f9612cfc198.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating IsClose_c727970057bb5b48903c1f9612cfc198 Done"
