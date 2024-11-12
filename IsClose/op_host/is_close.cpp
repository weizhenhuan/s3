
#include "is_close_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  IsCloseTilingData tiling;
  int32_t NUM = 8;
  uint32_t sizeofdatatype;
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint64_t ub_size;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  auto aivNum = ascendcPlatform.GetCoreNum();
  auto rtol = context->GetAttrs()->GetFloat(0);
  auto atol = context->GetAttrs()->GetFloat(1);
  auto equal_nan = context->GetAttrs()->GetBool(2);

  uint32_t totalLengthX1 =
      context->GetInputShape(0)->GetStorageShape().GetShapeSize();
  uint32_t totalLengthX2 =
      context->GetInputShape(1)->GetStorageShape().GetShapeSize();
  uint32_t totalLengthY =
      context->GetOutputShape(0)->GetStorageShape().GetShapeSize();

  auto dt = context->GetInputDesc(0)->GetDataType();
  if (dt == ge::DT_INT8) {
    sizeofdatatype = 1;
    NUM = 20;
  } else if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
    sizeofdatatype = 2;
    NUM = 12;
  } else {
    sizeofdatatype = 4;
    NUM = 8;
  }
    NUM =30;
  uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
  uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;
  tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;
  uint32_t block_size = tiling_size * ALIGN_NUM;
  aivNum = 1;

  tiling.set_totalLengthX1(totalLengthX1);
  tiling.set_totalLengthX2(totalLengthX2);
  tiling.set_totalLengthY(totalLengthY);
  tiling.set_ALIGN_NUM(ALIGN_NUM);
  tiling.set_tiling_size(tiling_size);
  tiling.set_block_size(block_size);
  tiling.set_aivNum(aivNum);
  tiling.set_rtol(*rtol);
  tiling.set_atol(*atol);
  tiling.set_equal_nan(*equal_nan);

  context->SetBlockDim(aivNum);
  printf(
      "########################################################################"
      "##\n");
  printf("                            THIS IS TILING DATAS\n");
  std::cout << "++ "
            << "ub_size = " << ub_size << "\n";
  std::cout << "++ "
            << "totalLengthX1 = " << totalLengthX1 << "\n";
  std::cout << "++ "
            << "totalLengthX2 = " << totalLengthX2 << "\n";
  std::cout << "++ "
            << "totalLengthY = " << totalLengthY << "\n";
  std::cout << "++ "
            << "rtol = " << *rtol << "\n";
  std::cout << "++ "
            << "atol = " << *atol << "\n";
  std::cout << "++ "
            << "equal_nan = " << *equal_nan << "\n";
  std::cout << "++ "
            << "ALIGN_NUM = " << ALIGN_NUM << "\n";
  std::cout << "++ "
            << "tiling_size = " << tiling_size << "\n";
  std::cout << "++ "
            << "block_size = " << block_size << "\n";
  std::cout << "++ "
            << "aivNum = " << aivNum << "\n";
  printf(
      "########################################################################"
      "##\n");
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t* currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class IsClose : public OpDef {
public:
    explicit IsClose(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("rtol").AttrType(OPTIONAL).Float(1e-05);
        this->Attr("atol").AttrType(OPTIONAL).Float(1e-08);
        this->Attr("equal_nan").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(IsClose);
}
