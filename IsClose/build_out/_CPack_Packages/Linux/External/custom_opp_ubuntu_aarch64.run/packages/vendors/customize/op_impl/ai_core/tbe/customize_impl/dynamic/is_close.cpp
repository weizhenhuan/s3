#include <type_traits>

#include "kernel_operator.h"
    using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X1, typename TYPE_X2, typename TYPE_Y>
class IsClose {
  using T = TYPE_X1;

 public:
  __aicore__ inline IsClose() {}
  __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, float rtol,
                              float atol, bool equal_nan, uint32_t ALIGN_NUM,
                              uint32_t block_size, uint32_t totalLengthX,
                              uint32_t totalLengthX2, uint32_t totalLengthY) {
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    this->tileLength = block_size;

    this->totalLengthX = totalLengthX;
    this->totalLengthY = totalLengthY;

    this->blockLengthX = totalLengthX;
    this->blockLengthY = totalLengthY;

    this->oriLengthX = this->blockLengthX;
    this->oriLengthY = this->blockLengthY;
    this->extraLengthX = this->blockLengthX % ALIGN_NUM
                             ? ALIGN_NUM - this->blockLengthX % ALIGN_NUM
                             : 0;
    this->extraLengthY = this->blockLengthY % ALIGN_NUM
                             ? ALIGN_NUM - this->blockLengthY % ALIGN_NUM
                             : 0;
    this->blockLengthX = this->blockLengthX + this->extraLengthX;  // 11472
    this->blockLengthY = this->blockLengthY + this->extraLengthY;  // 11472

    this->rtol = rtol;
    this->atol = atol;
    this->equal_nan = equal_nan;

    auto bufferlengthX = this->blockLengthX;
    auto bufferlengthY = this->blockLengthY;

    Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1, bufferlengthX);
    Gm_x2.SetGlobalBuffer((__gm__ TYPE_X1*)x2, bufferlengthX);
    Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y,
                         bufferlengthY);  

    this->tileNumX = this->blockLengthX / this->tileLength +
                     (this->blockLengthX % this->tileLength > 0);

    pipe.InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1));
    pipe.InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1));
    pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));

    // 初始化中间计算所需的buffer
    pipe.InitBuffer(B_diff, this->tileLength * sizeof(float));
    pipe.InitBuffer(B_bits, this->tileLength * sizeof(uint8_t));
    pipe.InitBuffer(B_result, this->tileLength * sizeof(half));

    pipe.InitBuffer(B_ones, this->tileLength * sizeof(half));
    this->ones = B_ones.Get<half>();
    Duplicate(this->ones, half(1), this->tileLength);
    if (std::is_same_v<T, int8_t>) {
      pipe.InitBuffer(B_x1, this->tileLength * sizeof(half));
      pipe.InitBuffer(B_x2, this->tileLength * sizeof(half));
    } else {
      pipe.InitBuffer(B_x1, this->tileLength * sizeof(float));
      pipe.InitBuffer(B_x2, this->tileLength * sizeof(float));
    }
  }

  __aicore__ inline void Process() {
    int32_t loopCountX = this->tileNumX;
    for (int32_t i = 0; i < loopCountX - 1; i++) {
      CopyIn(i, this->tileLength);
      Compute(i, this->tileLength);
      CopyOut(i, this->tileLength);
    }
    auto lengthX = this->blockLengthX - this->tileLength * (loopCountX - 1);
    auto cacLengthX = this->oriLengthX - this->tileLength * (loopCountX - 1);
    CopyIn(loopCountX - 1, lengthX);
    Compute(loopCountX - 1, cacLengthX);  // indicies存储于transferBuffer
    CopyOut(loopCountX - 1, lengthX);
  }

 private:
  __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
    LocalTensor<TYPE_X1> x1Local = Q_x1.AllocTensor<TYPE_X1>();
    LocalTensor<TYPE_X1> x2Local = Q_x2.AllocTensor<TYPE_X1>();
    DataCopy(x1Local, Gm_x1[progress * this->tileLength], length);
    DataCopy(x2Local, Gm_x2[progress * this->tileLength], length);
    Q_x1.EnQue(x1Local);
    Q_x2.EnQue(x2Local);
  }

  __aicore__ inline void Compute(int32_t progress, uint32_t length) {
    LocalTensor<TYPE_X1> x1 = Q_x1.DeQue<TYPE_X1>();
    LocalTensor<TYPE_X1> x2 = Q_x2.DeQue<TYPE_X1>();
    LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();

    auto bits = B_bits.Get<uint8_t>();
    auto result = B_result.Get<half>();
    auto inty = y.template ReinterpretCast<uint8_t>();

    if constexpr (std::is_same_v<T, int8_t> ||
                  std::is_same_v<T, unsigned char>) {
      auto half_x1 = B_x1.Get<half>();
      auto half_x2 = B_x2.Get<half>();
      Cast(half_x1, x1, RoundMode::CAST_NONE, length);
      Cast(half_x2, x2, RoundMode::CAST_NONE, length);

      // 计算|x1 - x2|
      Sub(half_x1, half_x1, half_x2, length);
      Abs(half_x1, half_x1, length);

      // 计算rtol * |x2| + atol
      Abs(half_x2, half_x2, length);
      Muls(half_x2, half_x2, (half)this->rtol, length);
      Adds(half_x2, half_x2, (half)this->atol, length);

      // 比较|x1 - x2| <= rtol * |x2| + atol
      Compare(bits, half_x1, half_x2, CMPMODE::LE, length);
    } else {
    //   对于float, int32_t, half类型的处理
      Sub(x1, x1, x2, length);
      Abs(x1, x1, length);

      Abs(x2, x2, length);
      Muls(x2, x2, (T)this->rtol, length);
      Adds(x2, x2, (T)this->atol, length);

      if constexpr (std::is_same_v<T, int32_t>) {
        auto val = B_x1.Get<float>();
        auto val2 = B_x2.Get<float>();
        Cast(val, x1, RoundMode::CAST_NONE, length);
        Cast(val2, x2, RoundMode::CAST_NONE, length);
        Compare(bits, val, val2, CMPMODE::LE, length);
      } else {  // float half
        Compare(bits, x1, x2, CMPMODE::LE, length);
      }
    }
    // 转换结果到输出类型
    Select(result, bits, this->ones, half(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
           length);
    Cast(inty, result, RoundMode::CAST_ROUND, length);

    Q_x1.FreeTensor(x1);
    Q_x2.FreeTensor(x2);
    Q_y.EnQue<TYPE_Y>(y);
  }

  __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
    LocalTensor<TYPE_Y> yLocal = Q_y.DeQue<TYPE_Y>();
    DataCopy(Gm_y[progress * this->tileLength], yLocal, length);
    Q_y.FreeTensor(yLocal);
  }

 private:
  TPipe pipe;
  TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1, Q_x2;
  TBuf<QuePosition::VECCALC> B_bits, B_result, B_ones, B_diff;
  TBuf<QuePosition::VECCALC> B_x1, B_x2;
  TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;

  GlobalTensor<TYPE_X1> Gm_x1;
  GlobalTensor<TYPE_X1> Gm_x2;
  GlobalTensor<TYPE_Y> Gm_y;
  LocalTensor<half> ones;

  uint32_t tileLength;
  uint32_t ALIGN_NUM;
  uint32_t tileNumX;
  uint32_t totalLengthX;
  uint32_t totalLengthY;
  uint32_t blockLengthX;
  uint32_t blockLengthY;
  uint32_t extraLengthX;
  uint32_t extraLengthY;
  uint32_t oriLengthX;
  uint32_t oriLengthY;

  float rtol;
  float atol;
  bool equal_nan;
};

extern "C" __global__ __aicore__ void is_close(GM_ADDR x1, GM_ADDR x2,
                                               GM_ADDR y, GM_ADDR workspace,
                                               GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  IsClose<DTYPE_X1, DTYPE_X2, DTYPE_Y> op;
  op.Init(x1, x2, y, tiling_data.rtol, tiling_data.atol, tiling_data.equal_nan,
          tiling_data.ALIGN_NUM, tiling_data.block_size,
          tiling_data.totalLengthX1, tiling_data.totalLengthX2,
          tiling_data.totalLengthY);
  op.Process();
}