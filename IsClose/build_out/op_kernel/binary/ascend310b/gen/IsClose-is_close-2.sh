#!/bin/bash
echo "[Ascend310B1] Generating IsClose_4e747a6b9e48ca8bb060ddb930c5080b ..."
opc $1 --main_func=is_close --input_param=/home/HwHiAiUser/s3/IsClose/build_out/op_kernel/binary/ascend310b/gen/IsClose_4e747a6b9e48ca8bb060ddb930c5080b_param.json --soc_version=Ascend310B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/IsClose_4e747a6b9e48ca8bb060ddb930c5080b.json ; then
  echo "$2/IsClose_4e747a6b9e48ca8bb060ddb930c5080b.json not generated!"
  exit 1
fi

if ! test -f $2/IsClose_4e747a6b9e48ca8bb060ddb930c5080b.o ; then
  echo "$2/IsClose_4e747a6b9e48ca8bb060ddb930c5080b.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating IsClose_4e747a6b9e48ca8bb060ddb930c5080b Done"
