#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IsCloseTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLengthX1);
TILING_DATA_FIELD_DEF(uint32_t, totalLengthX2);
TILING_DATA_FIELD_DEF(uint32_t, totalLengthY);
TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
TILING_DATA_FIELD_DEF(uint32_t, tiling_size);
TILING_DATA_FIELD_DEF(uint32_t, block_size);
TILING_DATA_FIELD_DEF(uint32_t, aivNum);
TILING_DATA_FIELD_DEF(float, rtol);
TILING_DATA_FIELD_DEF(float, atol);
TILING_DATA_FIELD_DEF(bool, equal_nan);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IsClose, IsCloseTilingData)
}  // namespace optiling
