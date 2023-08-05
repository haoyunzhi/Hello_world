#include <immintrin.h> //AVX AVX2
#include <stdio.h>
#include <stdlib.h> //malloc+free
#include "FUNCTION.h"
#include <wmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

//重复加密轮
//Round0
//Round1
//Round2
//Round3

//注意函数形参
unsigned int SM4_GCMPLUS(__m128i IV, __m256i NONCE96[32], __m256i key[32][8], __m256i over[32], unsigned long ulInitialInput[4], unsigned long ulKeyList1[32], unsigned long ulKeyList2[32])
{
    //Packing...
    //Seperate bits for S-box
    for (int i = 0; i < 4; i++)
    {
        SWAPMOVE(NONCE96[8 * i + 0], NONCE96[8 * i + 1], MASK1, 1);
        SWAPMOVE(NONCE96[8 * i + 2], NONCE96[8 * i + 3], MASK1, 1);
        SWAPMOVE(NONCE96[8 * i + 4], NONCE96[8 * i + 5], MASK1, 1);
        SWAPMOVE(NONCE96[8 * i + 6], NONCE96[8 * i + 7], MASK1, 1);

        SWAPMOVE(NONCE96[8 * i + 0], NONCE96[8 * i + 2], MASK2, 2);
        SWAPMOVE(NONCE96[8 * i + 1], NONCE96[8 * i + 3], MASK2, 2);
        SWAPMOVE(NONCE96[8 * i + 4], NONCE96[8 * i + 6], MASK2, 2);
        SWAPMOVE(NONCE96[8 * i + 5], NONCE96[8 * i + 7], MASK2, 2);

        SWAPMOVE(NONCE96[8 * i + 0], NONCE96[8 * i + 4], MASK4, 4);
        SWAPMOVE(NONCE96[8 * i + 1], NONCE96[8 * i + 5], MASK4, 4);
        SWAPMOVE(NONCE96[8 * i + 2], NONCE96[8 * i + 6], MASK4, 4);
        SWAPMOVE(NONCE96[8 * i + 3], NONCE96[8 * i + 7], MASK4, 4);
    }
    //Group the rows for efficient MixColumns implementation
    for (int i = 0; i < 8; i++)
    {
        SWAPMOVE(NONCE96[i + 8], NONCE96[i + 0], MASK32, 32);
        SWAPMOVE(NONCE96[i + 24], NONCE96[i + 16], MASK32, 32);

        SWAPMOVEBY64(NONCE96[i + 16], NONCE96[i + 0], MASK64);
        SWAPMOVEBY64(NONCE96[i + 24], NONCE96[i + 8], MASK64);
    }
    //high->low (x[0]->x[7])
    __m256i pt[32];
    for (int i = 0; i < 8; i++)
    {
        pt[7 - i] = NONCE96[i];
        pt[15 - i] = NONCE96[i + 8];
        pt[23 - i] = NONCE96[i + 16];
        pt[31 - i] = NONCE96[i + 24];
    }//high->low (pt[7]->pt[0])
    //此处以上得到需要进行以bitslice形式进行SMS4加密的pt[0]-pt[31]

    //SM4共32轮 (ROUND0-ROUND31)
    __m256i* point;//pt[0]-pt[31]赋值给point[0]-point[31]
    //point指针接收pt[0]-pt[31]的传值
    point = ENC_ROUND0(pt, key[0]);
    point = ENC_ROUND1(point, key[1]);
    point = ENC_ROUND2(point, key[2]);
    point = ENC_ROUND3(point, key[3]);

    point = ENC_ROUND0(point, key[4]);
    point = ENC_ROUND1(point, key[5]);
    point = ENC_ROUND2(point, key[6]);
    point = ENC_ROUND3(point, key[7]);

    point = ENC_ROUND0(point, key[8]);
    point = ENC_ROUND1(point, key[9]);
    point = ENC_ROUND2(point, key[10]);
    point = ENC_ROUND3(point, key[11]);

    point = ENC_ROUND0(point, key[12]);
    point = ENC_ROUND1(point, key[13]);
    point = ENC_ROUND2(point, key[14]);
    point = ENC_ROUND3(point, key[15]);

    point = ENC_ROUND0(point, key[16]);
    point = ENC_ROUND1(point, key[17]);
    point = ENC_ROUND2(point, key[18]);
    point = ENC_ROUND3(point, key[19]);

    point = ENC_ROUND0(point, key[20]);
    point = ENC_ROUND1(point, key[21]);
    point = ENC_ROUND2(point, key[22]);
    point = ENC_ROUND3(point, key[23]);

    point = ENC_ROUND0(point, key[24]);
    point = ENC_ROUND1(point, key[25]);
    point = ENC_ROUND2(point, key[26]);
    point = ENC_ROUND3(point, key[27]);

    point = ENC_ROUND0(point, key[28]);
    point = ENC_ROUND1(point, key[29]);
    point = ENC_ROUND2(point, key[30]);
    point = ENC_ROUND3(point, key[31]);

    //反序输出point[0]~point[31]
    //定义1个256比特的数暂时存放交换的中间值
    __m256i tomato;
    //point[0]~point[7] 与 point[24]~point[31]
    for (int orange = 0; orange < 8; orange++)
    {
        tomato = point[24 + orange];
        point[24 + orange] = point[orange];
        point[orange] = tomato;
    }
    //point[8]~point[15] 与 point[16]~point[23]
    for (int orange = 0; orange < 8; orange++)
    {
        tomato = point[16 + orange];
        point[16 + orange] = point[8 + orange];
        point[8 + orange] = tomato;
    }

    //直接进行数组间的异或运算（密文与待加密的内容）
    //point[i] XOR over[i]
    __m256i ct[32];
    for (int i = 0; i < 32; i++)
        ct[i] = _mm256_xor_si256(point[i], over[i]);
    //定义拆分需要的128-bit数组split[64]
    __m128i split[64];
    for (int k = 0; k < 32; k++)
    {
        unsigned int* deliver = (unsigned int*)&ct[k];
        split[2 * k] = _mm_set_epi32(deliver[7], deliver[6], deliver[5], deliver[4]);
        split[2 * k + 1] = _mm_set_epi32(deliver[3], deliver[2], deliver[1], deliver[0]);
    }

    //为了符合GCM的形式需要且仅需要NONCE96的SM4加密结果  
    unsigned long* non;
    non = SM4_LUT(ulKeyList1, ulInitialInput);
    __m128i ct0 = _mm_set_epi32(non[0], non[1], non[2], non[3]); //SM4查找表实现结果
    //H
    unsigned long* h;
    unsigned long H0[4] = { 0, 0, 0, 0 };
    h = SM4_LUT(ulKeyList2, H0);
    __m128i H = _mm_set_epi32(h[0], h[1], h[2], h[3]); //SM4查找表实现结果


    //GCM
    //初始化全零的变量存储过程中需要传递的中间值
    __m128i Y = _mm_setzero_si128();
    //初始化全零的变量处理传递过来的中间值
    __m128i M = _mm_setzero_si128();
    //提前计算出H1、H2、H3、H4
    __m128i H1 = H;
    __m128i H2, H3, H4;
    __m128i h1, h2, h3, h4;
    //bit reflection
    h1 = Rbit128(H1); //h1 已完成 bit reflection
    gfmul(h1, h1, &h2);
    H2 = Rbit128(h2);
    gfmul(h1, h2, &h3);
    H3 = Rbit128(h3);
    gfmul(h1, h3, &h4);
    H4 = Rbit128(h4);

    //ct[i] 在做有限域乘法前先完成 bit reflection
    for (int i = 0; i < 64; i++)
        split[i] = Rbit128(split[i]);
    __m128i NONCE128_Rbit = Rbit128(IV);
    __m128i AAD; //假设此处的附加信息AAD是初始向量IV即nonce
    gfmul(NONCE128_Rbit, h1, &AAD);
    AAD = Rbit128(AAD);

    M = _mm_xor_si128(split[0], AAD); //均完成 bit reflection
    reduce4(h1, h2, h3, h4, split[3], split[2], split[1], M, &Y);
    M = _mm_xor_si128(split[4], Y);
    reduce4(h1, h2, h3, h4, split[7], split[6], split[5], M, &Y);
    M = _mm_xor_si128(split[8], Y);
    reduce4(h1, h2, h3, h4, split[11], split[10], split[9], M, &Y);
    M = _mm_xor_si128(split[12], Y);
    reduce4(h1, h2, h3, h4, split[15], split[14], split[13], M, &Y);

    M = _mm_xor_si128(split[16], Y);
    reduce4(h1, h2, h3, h4, split[19], split[18], split[17], M, &Y);
    M = _mm_xor_si128(split[20], Y);
    reduce4(h1, h2, h3, h4, split[23], split[22], split[21], M, &Y);
    M = _mm_xor_si128(split[24], Y);
    reduce4(h1, h2, h3, h4, split[27], split[26], split[25], M, &Y);
    M = _mm_xor_si128(split[28], Y);
    reduce4(h1, h2, h3, h4, split[31], split[30], split[29], M, &Y);

    M = _mm_xor_si128(split[32], Y);
    reduce4(h1, h2, h3, h4, split[35], split[34], split[33], M, &Y);
    M = _mm_xor_si128(split[36], Y);
    reduce4(h1, h2, h3, h4, split[39], split[38], split[37], M, &Y);
    M = _mm_xor_si128(split[40], Y);
    reduce4(h1, h2, h3, h4, split[43], split[42], split[41], M, &Y);
    M = _mm_xor_si128(split[44], Y);
    reduce4(h1, h2, h3, h4, split[47], split[46], split[45], M, &Y);

    M = _mm_xor_si128(split[48], Y);
    reduce4(h1, h2, h3, h4, split[51], split[50], split[49], M, &Y);
    M = _mm_xor_si128(split[52], Y);
    reduce4(h1, h2, h3, h4, split[55], split[54], split[53], M, &Y);
    M = _mm_xor_si128(split[56], Y);
    reduce4(h1, h2, h3, h4, split[59], split[58], split[57], M, &Y);
    M = _mm_xor_si128(split[60], Y);
    reduce4(h1, h2, h3, h4, split[63], split[62], split[61], M, &Y);

    Y = Rbit128(Y);

    //M = _mm_insert_epi32(M, 8064, 0);
    //M = _mm_insert_epi64(M, 128, 1);
    //指令 _mm_insert_epi64 (__m128i a, __int64 i, const int imm8) 编译时报错

    M = _mm_set_epi32(0x00000000, 0x00000080, 0x00000000, 0x00001F80);
    Y = _mm_xor_si128(M, Y);
    Y = Rbit128(Y);
    gfmul(h1, Y, &Y);
    Y = Rbit128(Y);
    __m128i tag = _mm_xor_si128(ct0, Y);


    //trick5：避免编译器优化掉 bit transformation 2
    __m128i and8 = _mm_set_epi32(0x00000000, 0x00000000, 0x00000000, 0xFF000000);
    Y = _mm_and_si128(Y, and8);
    unsigned int* extract = (unsigned int*)&Y;
    unsigned int back5 = extract[0];

    return back5;
}

