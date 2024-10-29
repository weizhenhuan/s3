#include <type_traits>
#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_Y, typename TYPE_DY, typename TYPE_Z>
class KernelAsinhGrad {
    using T = TYPE_Y;

public:
    __aicore__ inline KernelAsinhGrad() {}
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR dy, GM_ADDR z, uint32_t totalLength, 
                                uint32_t ALIGN_NUM, uint32_t block_size, 
                                uint32_t core_size, uint32_t core_remain) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        // printf("block num: %d, block id: %d", (int)GetBlockNum(), (int)GetBlockIdx());
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        // get start index for current core, core parallel
        dyGm.SetGlobalBuffer((__gm__ DTYPE_DY*)dy + startPointer, bufferlength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + startPointer, bufferlength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z*)z + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(inQueueDY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM,
                        this->tileLength * sizeof(DTYPE_Y));

        pipe.InitBuffer(B1, this->tileLength * sizeof(T));
        pipe.InitBuffer(B2, this->tileLength * sizeof(T));
        // pipe.InitBuffer(B_zeros, this->tileLength * sizeof(T));
        // this->zeros = B_zeros.Get<T>();
        // Duplicate(this->zeros, (T)0, this->tileLength);
        if constexpr (std::is_same_v<T, half>) {
            pipe.InitBuffer(B32_1, this->tileLength * sizeof(float));
            pipe.InitBuffer(B32_2, this->tileLength * sizeof(float));
        }
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        auto length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length);
        CopyOut(loopCount - 1, length);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        LocalTensor<DTYPE_Y> dyLocal = inQueueDY.AllocTensor<DTYPE_Y>();
        DataCopy(yLocal, yGm[progress * this->tileLength], length);
        DataCopy(dyLocal, dyGm[progress * this->tileLength], length);
        inQueueY.EnQue(yLocal);
        inQueueDY.EnQue(dyLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        LocalTensor<DTYPE_Y> dyLocal = inQueueDY.DeQue<DTYPE_Y>();
        LocalTensor<DTYPE_Y> zLocal = outQueueZ.AllocTensor<DTYPE_Y>();

        if constexpr (std::is_same_v<T, float>){
            auto b1 = B1.DeQue<T>();
            auto b2 = B2.DeQue<T>();
            Sub(b1, this->zeros, yLocal, length);
            Exp(b1, b1, length);
            Exp(b2, yLocal, length);
            Add(b1, b1, b2, length);
            Muls(b2, dyLocal, (T)2, length);
            Div(zLocal, b2, b1, length);
        } else {
          auto fp32 = B32_1.Get<float>();
          auto fp32_2 = B32_2.Get<float>();
          auto b1 = B1.DeQue<T>();

        //   Cast(fp32, yLocal, RoundMode::CAST_NONE, length); 
        //   Exp(fp32_2, fp32, length);   // e^x
        //   Muls(fp32_2, fp32_2, (float)2, length);  // e^2x
        //   Cast(fp32, dyLocal, RoundMode::CAST_NONE, length); 
        //   Mul(fp32_2, fp32, fp32_2, length); // 2 e^x dy
             
        //   Cast(fp32, yLocal, RoundMode::CAST_NONE, length); 
        //   Muls(fp32, fp32, (float)2, length);   // 2x
        //   Exp(fp32, fp32, length);   //e^2x
        //   Adds(fp32, fp32, (float)1, length); // e^2x + 1
        //   Div(fp32, fp32_2, fp32, length);
        //   Cast(zLocal, fp32, RoundMode::CAST_NONE, length); 
            Sub(b1, this->zeros, yLocal, length);
            Cast(fp32, b1, RoundMode::CAST_NONE, length);
            Exp(fp32, fp32, length);
            Cast(fp32_2, yLocal, RoundMode::CAST_NONE, length);
            Exp(fp32_2, fp32_2, length);
            Add(fp32, fp32, fp32_2, length);
            Cast(fp32_2, dyLocal, RoundMode::CAST_NONE, length);
            Muls(fp32_2, fp32_2, (float)2, length);
            Div(fp32, fp32_2, fp32, length);
            Cast(zLocal, fp32, RoundMode::CAST_NONE, length); 
        }

        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueY.FreeTensor(yLocal);
        inQueueDY.FreeTensor(dyLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<DTYPE_Y> zLocal = outQueueZ.DeQue<DTYPE_Y>();
        DataCopy(zGm[progress * this->tileLength], zLocal, length);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY, inQueueDY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    TBuf<QuePosition::VECCALC> B1, B2,B_zeros, B32_1, B32_2;

    GlobalTensor<DTYPE_Y> dyGm;
    GlobalTensor<DTYPE_Y> yGm;
    GlobalTensor<DTYPE_Y> zGm;
    LocalTensor<T> zeros;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void asinh_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelAsinhGrad<DTYPE_Y, DTYPE_DY, DTYPE_Z> op;
    op.Init(y, dy,z, tiling_data.totalLength, tiling_data.ALIGN_NUM, 
            tiling_data.block_size, tiling_data.core_size, 
            tiling_data.core_remain);
    op.Process();
}