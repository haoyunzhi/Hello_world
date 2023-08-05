#include <immintrin.h> //AVX AVX2
#include <stdio.h>
#include <time.h>
#include "FUNCTION.h"
//< > 该符号表示引用编译器的类库路径里面的头文件，" "表示引用同一文件夹下的头文件
#include <wmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include <windows.h>
#include <process.h>
#include<Psapi.h> //内存占用
#pragma comment(lib,"Psapi.lib")

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

inline
unsigned long long readTSC() {
    // _mm_lfence();  // optionally wait for earlier insns to retire before reading the clock
    return __rdtsc();
    // _mm_lfence();  // optionally block later instructions until rdtsc retires
}


int main()
{
    // (96-bit nonce) || (32-bit counter)
    //这里假定128bit的counter block初始值为 [0xFFFF0000 FFFF0000 FFFF0000 00000000] 
    __m128i IV = _mm_set_epi32(0xFFFF0000, 0xFFFF0000, 0xFFFF0000, 0x00000000);
    unsigned long ulInitialInput[4] = { 0xFFFF0000, 0xFFFF0000, 0xFFFF0000, 0x00000001 }; //NONCE96(128bit)
    unsigned long ulKeyList1[32] =
    { 0, 1, 2, 3, 4, 5, 6, 7,
      8, 9, 10, 11, 12, 13, 14, 15,
      16, 17, 18, 19, 20, 21, 22, 23,
      24, 25, 26, 27, 28, 29, 30, 31
    }; //加密ct0
    unsigned long ulKeyList2[32] =
    { 0, 1, 2, 3, 4, 5, 6, 7,
      8, 9, 10, 11, 12, 13, 14, 15,
      16, 17, 18, 19, 20, 21, 22, 23,
      24, 25, 26, 27, 28, 29, 30, 31
    }; //加密all zero string

    __m256i plaintext[32];//正常排序存放（按列）
    plaintext[0] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000000, 0xFFFF0000FFFF0000, 0xFFFF0000000000FF);
    plaintext[1] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000000, 0xFFFF0000FFFF0000, 0xFFFF000000000000);
    plaintext[2] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000000000FF, 0xFFFF0000FFFF0000, 0xFFFF0000000000FF);
    plaintext[3] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000000000FF, 0xFFFF0000FFFF0000, 0xFFFF000000000000);
    plaintext[4] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000000, 0xFFFF0000FFFF0000, 0xFFFF0000000000FF);
    plaintext[5] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000000, 0xFFFF0000FFFF0000, 0xFFFF000000000000);
    plaintext[6] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000000000FF, 0xFFFF0000FFFF0000, 0xFFFF0000000000FF);
    plaintext[7] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000000000FF, 0xFFFF0000FFFF0000, 0xFFFF000000000000);

    plaintext[8] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000000, 0xFFFF0000FFFF0000, 0xFFFF0000000000FF);
    plaintext[9] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000000, 0xFFFF0000FFFF0000, 0xFFFF000000000000);
    plaintext[10] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000000000FF, 0xFFFF0000FFFF0000, 0xFFFF0000000000FF);
    plaintext[11] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000000000FF, 0xFFFF0000FFFF0000, 0xFFFF000000000000);
    plaintext[12] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000000, 0xFFFF0000FFFF0000, 0xFFFF0000000000FF);
    plaintext[13] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000000, 0xFFFF0000FFFF0000, 0xFFFF000000000000);
    plaintext[14] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000000000FF, 0xFFFF0000FFFF0000, 0xFFFF0000000000FF);
    plaintext[15] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000000000FF, 0xFFFF0000FFFF0000, 0xFFFF000000000000);

    plaintext[16] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000000, 0xFFFF0000FFFF0000, 0xFFFF0000000000FF);
    plaintext[17] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000000, 0xFFFF0000FFFF0000, 0xFFFF000000000000);
    plaintext[18] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000000000FF, 0xFFFF0000FFFF0000, 0xFFFF0000000000FF);
    plaintext[19] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000000000FF, 0xFFFF0000FFFF0000, 0xFFFF000000000000);
    plaintext[20] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000000, 0xFFFF0000FFFF0000, 0xFFFF0000000000FF);
    plaintext[21] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000000, 0xFFFF0000FFFF0000, 0xFFFF000000000000);
    plaintext[22] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000000000FF, 0xFFFF0000FFFF0000, 0xFFFF0000000000FF);
    plaintext[23] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000000000FF, 0xFFFF0000FFFF0000, 0xFFFF000000000000);

    plaintext[24] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000000, 0xFFFF0000FFFF0000, 0xFFFF0000000000FF);
    plaintext[25] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000000, 0xFFFF0000FFFF0000, 0xFFFF000000000000);
    plaintext[26] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000000000FF, 0xFFFF0000FFFF0000, 0xFFFF0000000000FF);
    plaintext[27] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000000000FF, 0xFFFF0000FFFF0000, 0xFFFF000000000000);
    plaintext[28] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000000, 0xFFFF0000FFFF0000, 0xFFFF0000000000FF);
    plaintext[29] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000000, 0xFFFF0000FFFF0000, 0xFFFF000000000000);
    plaintext[30] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000000000FF, 0xFFFF0000FFFF0000, 0xFFFF0000000000FF);
    plaintext[31] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000000000FF, 0xFFFF0000FFFF0000, 0xFFFF000000000000);

    __m256i NONCE96[32];//正常排序存放（按列）
    NONCE96[0] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000002, 0xFFFF0000FFFF0000, 0xFFFF000000000003);
    NONCE96[1] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000004, 0xFFFF0000FFFF0000, 0xFFFF000000000005);
    NONCE96[2] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000006, 0xFFFF0000FFFF0000, 0xFFFF000000000007);
    NONCE96[3] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000008, 0xFFFF0000FFFF0000, 0xFFFF000000000009);
    NONCE96[4] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF00000000000A, 0xFFFF0000FFFF0000, 0xFFFF00000000000B);
    NONCE96[5] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF00000000000C, 0xFFFF0000FFFF0000, 0xFFFF00000000000D);
    NONCE96[6] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF00000000000E, 0xFFFF0000FFFF0000, 0xFFFF00000000000F);
    NONCE96[7] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000010, 0xFFFF0000FFFF0000, 0xFFFF000000000011);

    NONCE96[8] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000012, 0xFFFF0000FFFF0000, 0xFFFF000000000013);
    NONCE96[9] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000014, 0xFFFF0000FFFF0000, 0xFFFF000000000015);
    NONCE96[10] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000016, 0xFFFF0000FFFF0000, 0xFFFF000000000017);
    NONCE96[11] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000018, 0xFFFF0000FFFF0000, 0xFFFF000000000019);
    NONCE96[12] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF00000000001A, 0xFFFF0000FFFF0000, 0xFFFF00000000001B);
    NONCE96[13] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF00000000001C, 0xFFFF0000FFFF0000, 0xFFFF00000000001D);
    NONCE96[14] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF00000000001E, 0xFFFF0000FFFF0000, 0xFFFF00000000001F);
    NONCE96[15] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000020, 0xFFFF0000FFFF0000, 0xFFFF000000000021);

    NONCE96[16] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000022, 0xFFFF0000FFFF0000, 0xFFFF000000000023);
    NONCE96[17] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000024, 0xFFFF0000FFFF0000, 0xFFFF000000000025);
    NONCE96[18] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000026, 0xFFFF0000FFFF0000, 0xFFFF000000000027);
    NONCE96[19] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000028, 0xFFFF0000FFFF0000, 0xFFFF000000000029);
    NONCE96[20] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF00000000002A, 0xFFFF0000FFFF0000, 0xFFFF00000000002B);
    NONCE96[21] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF00000000002C, 0xFFFF0000FFFF0000, 0xFFFF00000000002D);
    NONCE96[22] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF00000000002E, 0xFFFF0000FFFF0000, 0xFFFF00000000002F);
    NONCE96[23] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000030, 0xFFFF0000FFFF0000, 0xFFFF000000000031);

    NONCE96[24] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000032, 0xFFFF0000FFFF0000, 0xFFFF000000000033);
    NONCE96[25] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000034, 0xFFFF0000FFFF0000, 0xFFFF000000000035);
    NONCE96[26] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000036, 0xFFFF0000FFFF0000, 0xFFFF000000000037);
    NONCE96[27] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000038, 0xFFFF0000FFFF0000, 0xFFFF000000000039);
    NONCE96[28] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF00000000003A, 0xFFFF0000FFFF0000, 0xFFFF00000000003B);
    NONCE96[29] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF00000000003C, 0xFFFF0000FFFF0000, 0xFFFF00000000003D);
    NONCE96[30] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF00000000003E, 0xFFFF0000FFFF0000, 0xFFFF00000000003F);
    NONCE96[31] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF000000000040, 0xFFFF0000FFFF0000, 0xFFFF000000000041);

    //待加密内容
    __m256i over[32]; //(standard)
    over[0] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[1] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[2] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[3] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[4] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[5] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[6] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[7] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);

    over[8] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[9] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[10] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[11] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[12] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[13] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[14] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[15] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);

    over[16] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[17] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[18] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[19] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[20] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[21] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[22] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[23] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);

    over[24] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[25] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[26] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[27] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[28] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[29] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[30] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);
    over[31] = _mm256_set_epi64x(0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000);

    //标准形式的分组也使用256bit的数据类型存放：block0(128bit) || block1(128bit)
    __m256i zero[32]; //定义加密内容数组(bitslice)
    //__m256i _mm256_set_epi64x(__int64 e3, __int64 e2, __int64 e1, __int64 e0)
    zero[0] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[1] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[2] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[3] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[4] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[5] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[6] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[7] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);

    zero[8] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[9] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[10] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[11] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[12] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[13] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[14] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[15] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);

    zero[16] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[17] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[18] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[19] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[20] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[21] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[22] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[23] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);

    zero[24] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[25] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[26] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[27] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[28] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[29] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[30] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
    zero[31] = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000);


    //设置轮密钥key[32][8]（任意(bitslice)）
    __m256i key[32][8];
    key[0][0] = _mm256_set_epi64x(0, 0, 0, 8);
    key[0][1] = _mm256_set_epi64x(0, 0, 0, 8);
    key[0][2] = _mm256_set_epi64x(0, 0, 0, 8);
    key[0][3] = _mm256_set_epi64x(0, 0, 0, 8);
    key[0][4] = _mm256_set_epi64x(0, 0, 0, 8);
    key[0][5] = _mm256_set_epi64x(0, 0, 0, 8);
    key[0][6] = _mm256_set_epi64x(0, 0, 0, 8);
    key[0][7] = _mm256_set_epi64x(0, 0, 0, 8);
    key[1][0] = _mm256_set_epi64x(0, 0, 0, 1);
    key[1][1] = _mm256_set_epi64x(0, 0, 0, 1);
    key[1][2] = _mm256_set_epi64x(0, 0, 0, 1);
    key[1][3] = _mm256_set_epi64x(0, 0, 0, 1);
    key[1][4] = _mm256_set_epi64x(0, 0, 0, 1);
    key[1][5] = _mm256_set_epi64x(0, 0, 0, 1);
    key[1][6] = _mm256_set_epi64x(0, 0, 0, 1);
    key[1][7] = _mm256_set_epi64x(0, 0, 0, 1);
    key[2][0] = _mm256_set_epi64x(0, 0, 0, 2);
    key[2][1] = _mm256_set_epi64x(0, 0, 0, 2);
    key[2][2] = _mm256_set_epi64x(0, 0, 0, 2);
    key[2][3] = _mm256_set_epi64x(0, 0, 0, 2);
    key[2][4] = _mm256_set_epi64x(0, 0, 0, 2);
    key[2][5] = _mm256_set_epi64x(0, 0, 0, 2);
    key[2][6] = _mm256_set_epi64x(0, 0, 0, 2);
    key[2][7] = _mm256_set_epi64x(0, 0, 0, 2);
    key[3][0] = _mm256_set_epi64x(0, 0, 0, 3);
    key[3][1] = _mm256_set_epi64x(0, 0, 0, 3);
    key[3][2] = _mm256_set_epi64x(0, 0, 0, 3);
    key[3][3] = _mm256_set_epi64x(0, 0, 0, 3);
    key[3][4] = _mm256_set_epi64x(0, 0, 0, 3);
    key[3][5] = _mm256_set_epi64x(0, 0, 0, 3);
    key[3][6] = _mm256_set_epi64x(0, 0, 0, 3);
    key[3][7] = _mm256_set_epi64x(0, 0, 0, 3);

    key[4][0] = _mm256_set_epi64x(0, 0, 0, 4);
    key[4][1] = _mm256_set_epi64x(0, 0, 0, 4);
    key[4][2] = _mm256_set_epi64x(0, 0, 0, 4);
    key[4][3] = _mm256_set_epi64x(0, 0, 0, 4);
    key[4][4] = _mm256_set_epi64x(0, 0, 0, 4);
    key[4][5] = _mm256_set_epi64x(0, 0, 0, 4);
    key[4][6] = _mm256_set_epi64x(0, 0, 0, 4);
    key[4][7] = _mm256_set_epi64x(0, 0, 0, 4);
    key[5][0] = _mm256_set_epi64x(0, 0, 0, 5);
    key[5][1] = _mm256_set_epi64x(0, 0, 0, 5);
    key[5][2] = _mm256_set_epi64x(0, 0, 0, 5);
    key[5][3] = _mm256_set_epi64x(0, 0, 0, 5);
    key[5][4] = _mm256_set_epi64x(0, 0, 0, 5);
    key[5][5] = _mm256_set_epi64x(0, 0, 0, 5);
    key[5][6] = _mm256_set_epi64x(0, 0, 0, 5);
    key[5][7] = _mm256_set_epi64x(0, 0, 0, 5);
    key[6][0] = _mm256_set_epi64x(0, 0, 0, 6);
    key[6][1] = _mm256_set_epi64x(0, 0, 0, 6);
    key[6][2] = _mm256_set_epi64x(0, 0, 0, 6);
    key[6][3] = _mm256_set_epi64x(0, 0, 0, 6);
    key[6][4] = _mm256_set_epi64x(0, 0, 0, 6);
    key[6][5] = _mm256_set_epi64x(0, 0, 0, 6);
    key[6][6] = _mm256_set_epi64x(0, 0, 0, 6);
    key[6][7] = _mm256_set_epi64x(0, 0, 0, 6);
    key[7][0] = _mm256_set_epi64x(0, 0, 0, 7);
    key[7][1] = _mm256_set_epi64x(0, 0, 0, 7);
    key[7][2] = _mm256_set_epi64x(0, 0, 0, 7);
    key[7][3] = _mm256_set_epi64x(0, 0, 0, 7);
    key[7][4] = _mm256_set_epi64x(0, 0, 0, 7);
    key[7][5] = _mm256_set_epi64x(0, 0, 0, 7);
    key[7][6] = _mm256_set_epi64x(0, 0, 0, 7);
    key[7][7] = _mm256_set_epi64x(0, 0, 0, 7);

    key[8][0] = _mm256_set_epi64x(0, 0, 1, 0);
    key[8][1] = _mm256_set_epi64x(0, 0, 1, 0);
    key[8][2] = _mm256_set_epi64x(0, 0, 1, 0);
    key[8][3] = _mm256_set_epi64x(0, 0, 1, 0);
    key[8][4] = _mm256_set_epi64x(0, 0, 1, 0);
    key[8][5] = _mm256_set_epi64x(0, 0, 1, 0);
    key[8][6] = _mm256_set_epi64x(0, 0, 1, 0);
    key[8][7] = _mm256_set_epi64x(0, 0, 1, 0);
    key[9][0] = _mm256_set_epi64x(0, 0, 2, 0);
    key[9][1] = _mm256_set_epi64x(0, 0, 2, 0);
    key[9][2] = _mm256_set_epi64x(0, 0, 2, 0);
    key[9][3] = _mm256_set_epi64x(0, 0, 2, 0);
    key[9][4] = _mm256_set_epi64x(0, 0, 2, 0);
    key[9][5] = _mm256_set_epi64x(0, 0, 2, 0);
    key[9][6] = _mm256_set_epi64x(0, 0, 2, 0);
    key[9][7] = _mm256_set_epi64x(0, 0, 2, 0);
    key[10][0] = _mm256_set_epi64x(0, 0, 3, 0);
    key[10][1] = _mm256_set_epi64x(0, 0, 3, 0);
    key[10][2] = _mm256_set_epi64x(0, 0, 3, 0);
    key[10][3] = _mm256_set_epi64x(0, 0, 3, 0);
    key[10][4] = _mm256_set_epi64x(0, 0, 3, 0);
    key[10][5] = _mm256_set_epi64x(0, 0, 3, 0);
    key[10][6] = _mm256_set_epi64x(0, 0, 3, 0);
    key[10][7] = _mm256_set_epi64x(0, 0, 3, 0);
    key[11][0] = _mm256_set_epi64x(0, 0, 4, 0);
    key[11][1] = _mm256_set_epi64x(0, 0, 4, 0);
    key[11][2] = _mm256_set_epi64x(0, 0, 4, 0);
    key[11][3] = _mm256_set_epi64x(0, 0, 4, 0);
    key[11][4] = _mm256_set_epi64x(0, 0, 4, 0);
    key[11][5] = _mm256_set_epi64x(0, 0, 4, 0);
    key[11][6] = _mm256_set_epi64x(0, 0, 4, 0);
    key[11][7] = _mm256_set_epi64x(0, 0, 4, 0);

    key[12][0] = _mm256_set_epi64x(0, 0, 5, 0);
    key[12][1] = _mm256_set_epi64x(0, 0, 5, 0);
    key[12][2] = _mm256_set_epi64x(0, 0, 5, 0);
    key[12][3] = _mm256_set_epi64x(0, 0, 5, 0);
    key[12][4] = _mm256_set_epi64x(0, 0, 5, 0);
    key[12][5] = _mm256_set_epi64x(0, 0, 5, 0);
    key[12][6] = _mm256_set_epi64x(0, 0, 5, 0);
    key[12][7] = _mm256_set_epi64x(0, 0, 5, 0);
    key[13][0] = _mm256_set_epi64x(0, 0, 6, 0);
    key[13][1] = _mm256_set_epi64x(0, 0, 6, 0);
    key[13][2] = _mm256_set_epi64x(0, 0, 6, 0);
    key[13][3] = _mm256_set_epi64x(0, 0, 6, 0);
    key[13][4] = _mm256_set_epi64x(0, 0, 6, 0);
    key[13][5] = _mm256_set_epi64x(0, 0, 6, 0);
    key[13][6] = _mm256_set_epi64x(0, 0, 6, 0);
    key[13][7] = _mm256_set_epi64x(0, 0, 6, 0);
    key[14][0] = _mm256_set_epi64x(0, 0, 7, 0);
    key[14][1] = _mm256_set_epi64x(0, 0, 7, 0);
    key[14][2] = _mm256_set_epi64x(0, 0, 7, 0);
    key[14][3] = _mm256_set_epi64x(0, 0, 7, 0);
    key[14][4] = _mm256_set_epi64x(0, 0, 7, 0);
    key[14][5] = _mm256_set_epi64x(0, 0, 7, 0);
    key[14][6] = _mm256_set_epi64x(0, 0, 7, 0);
    key[14][7] = _mm256_set_epi64x(0, 0, 7, 0);
    key[15][0] = _mm256_set_epi64x(0, 0, 8, 0);
    key[15][1] = _mm256_set_epi64x(0, 0, 8, 0);
    key[15][2] = _mm256_set_epi64x(0, 0, 8, 0);
    key[15][3] = _mm256_set_epi64x(0, 0, 8, 0);
    key[15][4] = _mm256_set_epi64x(0, 0, 8, 0);
    key[15][5] = _mm256_set_epi64x(0, 0, 8, 0);
    key[15][6] = _mm256_set_epi64x(0, 0, 8, 0);
    key[15][7] = _mm256_set_epi64x(0, 0, 8, 0);

    key[16][0] = _mm256_set_epi64x(0, 1, 0, 0);
    key[16][1] = _mm256_set_epi64x(0, 1, 0, 0);
    key[16][2] = _mm256_set_epi64x(0, 1, 0, 0);
    key[16][3] = _mm256_set_epi64x(0, 1, 0, 0);
    key[16][4] = _mm256_set_epi64x(0, 1, 0, 0);
    key[16][5] = _mm256_set_epi64x(0, 1, 0, 0);
    key[16][6] = _mm256_set_epi64x(0, 1, 0, 0);
    key[16][7] = _mm256_set_epi64x(0, 1, 0, 0);
    key[17][0] = _mm256_set_epi64x(0, 2, 0, 0);
    key[17][1] = _mm256_set_epi64x(0, 2, 0, 0);
    key[17][2] = _mm256_set_epi64x(0, 2, 0, 0);
    key[17][3] = _mm256_set_epi64x(0, 2, 0, 0);
    key[17][4] = _mm256_set_epi64x(0, 2, 0, 0);
    key[17][5] = _mm256_set_epi64x(0, 2, 0, 0);
    key[17][6] = _mm256_set_epi64x(0, 2, 0, 0);
    key[17][7] = _mm256_set_epi64x(0, 2, 0, 0);
    key[18][0] = _mm256_set_epi64x(0, 3, 0, 0);
    key[18][1] = _mm256_set_epi64x(0, 3, 0, 0);
    key[18][2] = _mm256_set_epi64x(0, 3, 0, 0);
    key[18][3] = _mm256_set_epi64x(0, 3, 0, 0);
    key[18][4] = _mm256_set_epi64x(0, 3, 0, 0);
    key[18][5] = _mm256_set_epi64x(0, 3, 0, 0);
    key[18][6] = _mm256_set_epi64x(0, 3, 0, 0);
    key[18][7] = _mm256_set_epi64x(0, 3, 0, 0);
    key[19][0] = _mm256_set_epi64x(0, 4, 0, 0);
    key[19][1] = _mm256_set_epi64x(0, 4, 0, 0);
    key[19][2] = _mm256_set_epi64x(0, 4, 0, 0);
    key[19][3] = _mm256_set_epi64x(0, 4, 0, 0);
    key[19][4] = _mm256_set_epi64x(0, 4, 0, 0);
    key[19][5] = _mm256_set_epi64x(0, 4, 0, 0);
    key[19][6] = _mm256_set_epi64x(0, 4, 0, 0);
    key[19][7] = _mm256_set_epi64x(0, 4, 0, 0);

    key[20][0] = _mm256_set_epi64x(0, 5, 0, 0);
    key[20][1] = _mm256_set_epi64x(0, 5, 0, 0);
    key[20][2] = _mm256_set_epi64x(0, 5, 0, 0);
    key[20][3] = _mm256_set_epi64x(0, 5, 0, 0);
    key[20][4] = _mm256_set_epi64x(0, 5, 0, 0);
    key[20][5] = _mm256_set_epi64x(0, 5, 0, 0);
    key[20][6] = _mm256_set_epi64x(0, 5, 0, 0);
    key[20][7] = _mm256_set_epi64x(0, 5, 0, 0);
    key[21][0] = _mm256_set_epi64x(0, 6, 0, 0);
    key[21][1] = _mm256_set_epi64x(0, 6, 0, 0);
    key[21][2] = _mm256_set_epi64x(0, 6, 0, 0);
    key[21][3] = _mm256_set_epi64x(0, 6, 0, 0);
    key[21][4] = _mm256_set_epi64x(0, 6, 0, 0);
    key[21][4] = _mm256_set_epi64x(0, 6, 0, 0);
    key[21][5] = _mm256_set_epi64x(0, 6, 0, 0);
    key[21][6] = _mm256_set_epi64x(0, 6, 0, 0);
    key[21][7] = _mm256_set_epi64x(0, 6, 0, 0);
    key[22][0] = _mm256_set_epi64x(0, 7, 0, 0);
    key[22][1] = _mm256_set_epi64x(0, 7, 0, 0);
    key[22][2] = _mm256_set_epi64x(0, 7, 0, 0);
    key[22][3] = _mm256_set_epi64x(0, 7, 0, 0);
    key[22][4] = _mm256_set_epi64x(0, 7, 0, 0);
    key[22][5] = _mm256_set_epi64x(0, 7, 0, 0);
    key[22][6] = _mm256_set_epi64x(0, 7, 0, 0);
    key[22][7] = _mm256_set_epi64x(0, 7, 0, 0);
    key[23][0] = _mm256_set_epi64x(0, 8, 0, 0);
    key[23][1] = _mm256_set_epi64x(0, 8, 0, 0);
    key[23][2] = _mm256_set_epi64x(0, 8, 0, 0);
    key[23][3] = _mm256_set_epi64x(0, 8, 0, 0);
    key[23][4] = _mm256_set_epi64x(0, 8, 0, 0);
    key[23][5] = _mm256_set_epi64x(0, 8, 0, 0);
    key[23][6] = _mm256_set_epi64x(0, 8, 0, 0);
    key[23][7] = _mm256_set_epi64x(0, 8, 0, 0);

    key[24][0] = _mm256_set_epi64x(1, 0, 0, 0);
    key[24][1] = _mm256_set_epi64x(1, 0, 0, 0);
    key[24][2] = _mm256_set_epi64x(1, 0, 0, 0);
    key[24][3] = _mm256_set_epi64x(1, 0, 0, 0);
    key[24][4] = _mm256_set_epi64x(1, 0, 0, 0);
    key[24][5] = _mm256_set_epi64x(1, 0, 0, 0);
    key[24][6] = _mm256_set_epi64x(1, 0, 0, 0);
    key[24][7] = _mm256_set_epi64x(1, 0, 0, 0);
    key[25][0] = _mm256_set_epi64x(2, 0, 0, 0);
    key[25][1] = _mm256_set_epi64x(2, 0, 0, 0);
    key[25][2] = _mm256_set_epi64x(2, 0, 0, 0);
    key[25][3] = _mm256_set_epi64x(2, 0, 0, 0);
    key[25][4] = _mm256_set_epi64x(2, 0, 0, 0);
    key[25][5] = _mm256_set_epi64x(2, 0, 0, 0);
    key[25][6] = _mm256_set_epi64x(2, 0, 0, 0);
    key[25][7] = _mm256_set_epi64x(2, 0, 0, 0);
    key[26][0] = _mm256_set_epi64x(3, 0, 0, 0);
    key[26][1] = _mm256_set_epi64x(3, 0, 0, 0);
    key[26][2] = _mm256_set_epi64x(3, 0, 0, 0);
    key[26][3] = _mm256_set_epi64x(3, 0, 0, 0);
    key[26][4] = _mm256_set_epi64x(3, 0, 0, 0);
    key[26][5] = _mm256_set_epi64x(3, 0, 0, 0);
    key[26][6] = _mm256_set_epi64x(3, 0, 0, 0);
    key[26][7] = _mm256_set_epi64x(3, 0, 0, 0);
    key[27][0] = _mm256_set_epi64x(4, 0, 0, 0);
    key[27][1] = _mm256_set_epi64x(4, 0, 0, 0);
    key[27][2] = _mm256_set_epi64x(4, 0, 0, 0);
    key[27][3] = _mm256_set_epi64x(4, 0, 0, 0);
    key[27][4] = _mm256_set_epi64x(4, 0, 0, 0);
    key[27][5] = _mm256_set_epi64x(4, 0, 0, 0);
    key[27][6] = _mm256_set_epi64x(4, 0, 0, 0);
    key[27][7] = _mm256_set_epi64x(4, 0, 0, 0);

    key[28][0] = _mm256_set_epi64x(5, 0, 0, 0);
    key[28][1] = _mm256_set_epi64x(5, 0, 0, 0);
    key[28][2] = _mm256_set_epi64x(5, 0, 0, 0);
    key[28][3] = _mm256_set_epi64x(5, 0, 0, 0);
    key[28][4] = _mm256_set_epi64x(5, 0, 0, 0);
    key[28][5] = _mm256_set_epi64x(5, 0, 0, 0);
    key[28][6] = _mm256_set_epi64x(5, 0, 0, 0);
    key[28][7] = _mm256_set_epi64x(5, 0, 0, 0);
    key[29][0] = _mm256_set_epi64x(6, 0, 0, 0);
    key[29][1] = _mm256_set_epi64x(6, 0, 0, 0);
    key[29][2] = _mm256_set_epi64x(6, 0, 0, 0);
    key[29][3] = _mm256_set_epi64x(6, 0, 0, 0);
    key[29][4] = _mm256_set_epi64x(6, 0, 0, 0);
    key[29][5] = _mm256_set_epi64x(6, 0, 0, 0);
    key[29][6] = _mm256_set_epi64x(6, 0, 0, 0);
    key[29][7] = _mm256_set_epi64x(6, 0, 0, 0);
    key[30][0] = _mm256_set_epi64x(7, 0, 0, 0);
    key[30][1] = _mm256_set_epi64x(7, 0, 0, 0);
    key[30][2] = _mm256_set_epi64x(7, 0, 0, 0);
    key[30][3] = _mm256_set_epi64x(7, 0, 0, 0);
    key[30][4] = _mm256_set_epi64x(7, 0, 0, 0);
    key[30][5] = _mm256_set_epi64x(7, 0, 0, 0);
    key[30][6] = _mm256_set_epi64x(7, 0, 0, 0);
    key[30][7] = _mm256_set_epi64x(7, 0, 0, 0);
    key[31][0] = _mm256_set_epi64x(8, 0, 0, 0);
    key[31][1] = _mm256_set_epi64x(8, 0, 0, 0);
    key[31][2] = _mm256_set_epi64x(8, 0, 0, 0);
    key[31][3] = _mm256_set_epi64x(8, 0, 0, 0);
    key[31][4] = _mm256_set_epi64x(8, 0, 0, 0);
    key[31][5] = _mm256_set_epi64x(8, 0, 0, 0);
    key[31][6] = _mm256_set_epi64x(8, 0, 0, 0);
    key[31][7] = _mm256_set_epi64x(8, 0, 0, 0);

    //有限域乘法的被乘数H（提前给出）   H=ENC(0^128)
    //__m128i H = _mm_set_epi32(0x0000FFFF, 0x0000FFFF, 0xFFFF0000, 0xFFFF0000);

    /*
    //TEST-CARRYLESSMULTIPLICATION
    __m128i xmm;
    __m128i xmm1 = _mm_set_epi32(0x7b5b5465, 0x73745665, 0x63746f72, 0x5d53475d);
    __m128i xmm2 = _mm_set_epi32(0x48692853, 0x68617929, 0x5b477565, 0x726f6e5d);
    xmm1 = Rbit128(xmm1);
    xmm2 = Rbit128(xmm2);
    gfmul(xmm1, xmm2, &xmm);
    xmm = Rbit128(xmm);
    unsigned int* op = (unsigned int*)&xmm; //output
    printf("%ud %ud %ud %ud \n\n", op[3], op[2], op[1], op[0]);
    */

    /*
    //test Rbit128();
    __m128i XXX = _mm_set_epi32(0x000000FF, 0xFF000000, 0x00000000, 0x00000000);
    __m128i XXXXX = Rbit128(XXX);
    unsigned int* OUT = (unsigned int*)&XXXXX;
    printf("%ud %ud %ud %ud \n\n", OUT[3], OUT[2], OUT[1], OUT[0]);
    */
    unsigned long long tsc1, tsc2;
    clock_t t1 = clock();
    tsc1 = readTSC();
    long int total_times = 2000; //尝试1000-2000

    long int sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0;
    unsigned int e0, e1, e2, e3, e4, e5;
    //都是用bitslice后的key
    for (int times = 0; times < total_times; times++)//103 104
    {
        //e0 = SM4(plaintext, key);
        //sum0 = sum0 + e0;
        
        //e1 = SM4PLUS(zero, key); 
        //sum1 = sum1 + e1;

        //e2 = SM4_CTR(NONCE96, key, over); 
        //sum2 = sum2 + e2;

        //e3 = SM4_CTRPLUS(NONCE96, key, over); 
        //sum3 = sum3 + e3;

        //e4 = SM4_GCM(IV, NONCE96, key, over, ulInitialInput, ulKeyList1, ulKeyList2); 
        //sum4 = sum4 + e4;

        e5 = SM4_GCMPLUS(IV, NONCE96, key, over, ulInitialInput, ulKeyList1, ulKeyList2); 
        sum5 = sum5 + e5;
    }
    tsc2 = readTSC();
    clock_t t2 = clock();
    printf(" %f        %f Gbps\n", (double)(t2 - t1) / CLOCKS_PER_SEC, (128.0 * total_times * 64) / ((double)(t2 - t1) / CLOCKS_PER_SEC) / 1073741824);// b/s 1024*1024*1024=1,073,741,824
    printf(" %lld cycles, %f bits per cycle, %f cycles per bytes", tsc2 - tsc1, (128.0 * total_times * 64) / ((double)(tsc2 - tsc1)), ((double)(tsc2 - tsc1)) / (16.0 * total_times * 64));
    printf("\n\n\n\n\n%ld\n", CLOCKS_PER_SEC);

    printf("\n\n%ld\n\n%ld\n\n%ld\n\n%ld\n\n%ld\n\n", sum1, sum2, sum3, sum4, sum5);

    HANDLE handle = GetCurrentProcess();
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(handle, &pmc, sizeof(pmc));
    printf("%d\r\n", pmc.WorkingSetSize); //结果保存单位是B，可以除以1000保存为kb格式
    

    return 0;
}
