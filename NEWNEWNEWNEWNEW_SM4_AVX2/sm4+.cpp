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
unsigned int SM4PLUS(__m256i pt[32], __m256i key[32][8])
{

    //加密全零串
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
    //至此仅完成SM4加密...

    //trick1：避免编译器优化
    __m256i and8 = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x00000000000000FF);
    point[7] = _mm256_and_si256(point[7], and8);
    unsigned int* extract = (unsigned int*)&point[7];
    unsigned int back1 = extract[0];
    //printf("\n\n %ud\n\n", extract[0]);
    return back1;
}

