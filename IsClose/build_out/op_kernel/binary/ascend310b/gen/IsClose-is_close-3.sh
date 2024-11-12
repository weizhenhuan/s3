#!/bin/bash
echo "[Ascend310B1] Generating IsClose_8bbdbcac6187ad53238e8958ca9755ac ..."
opc $1 --main_func=is_close --input_param=/home/HwHiAiUser/s3/IsClose/build_out/op_kernel/binary/ascend310b/gen/IsClose_8bbdbcac6187ad53238e8958ca9755ac_param.json --soc_version=Ascend310B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/IsClose_8bbdbcac6187ad53238e8958ca9755ac.json ; then
  echo "$2/IsClose_8bbdbcac6187ad53238e8958ca9755ac.json not generated!"
  exit 1
fi

if ! test -f $2/IsClose_8bbdbcac6187ad53238e8958ca9755ac.o ; then
  echo "$2/IsClose_8bbdbcac6187ad53238e8958ca9755ac.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating IsClose_8bbdbcac6187ad53238e8958ca9755ac Done"
