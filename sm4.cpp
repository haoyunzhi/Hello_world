#include <immintrin.h> //AVX AVX2
#include <stdio.h>
#include <stdlib.h> //malloc+free
#include "FUNCTION.h"
#include <wmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

//注意函数形参
//这里的形参取标准形式的明文分组...
unsigned int SM4(__m256i x[32], __m256i key[32][8])
{
    //Packing...
    //Seperate bits for S-box
    for (int i = 0; i < 4; i++)
    {
        SWAPMOVE(x[8 * i + 0], x[8 * i + 1], MASK1, 1);
        SWAPMOVE(x[8 * i + 2], x[8 * i + 3], MASK1, 1);
        SWAPMOVE(x[8 * i + 4], x[8 * i + 5], MASK1, 1);
        SWAPMOVE(x[8 * i + 6], x[8 * i + 7], MASK1, 1);

        SWAPMOVE(x[8 * i + 0], x[8 * i + 2], MASK2, 2);
        SWAPMOVE(x[8 * i + 1], x[8 * i + 3], MASK2, 2);
        SWAPMOVE(x[8 * i + 4], x[8 * i + 6], MASK2, 2);
        SWAPMOVE(x[8 * i + 5], x[8 * i + 7], MASK2, 2);

        SWAPMOVE(x[8 * i + 0], x[8 * i + 4], MASK4, 4);
        SWAPMOVE(x[8 * i + 1], x[8 * i + 5], MASK4, 4);
        SWAPMOVE(x[8 * i + 2], x[8 * i + 6], MASK4, 4);
        SWAPMOVE(x[8 * i + 3], x[8 * i + 7], MASK4, 4);
    }
    //Group the rows for efficient MixColumns implementation
    for (int i = 0; i < 8; i++)
    {
        SWAPMOVE(x[i + 8], x[i + 0], MASK32, 32);
        SWAPMOVE(x[i + 24], x[i + 16], MASK32, 32);

        SWAPMOVEBY64(x[i + 16], x[i + 0], MASK64);
        SWAPMOVEBY64(x[i + 24], x[i + 8], MASK64);
    }
    //high->low (x[0]->x[7])
    __m256i pt[32];
    for (int i = 0; i < 8; i++)
    {
        pt[7 - i] = x[i];
        pt[15 - i] = x[i + 8];
        pt[23 - i] = x[i + 16];
        pt[31 - i] = x[i + 24];
    }//high->low (pt[7]->pt[0])
        


    //SM4共32轮 (ROUND0-ROUND31)
    //重复加密轮
    //Round0
    //Round1
    //Round2
    //Round3
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


    
    //high->low (point[7]->point[0])
    for (int i = 0; i < 8; i++)
    {
        x[i] = point[7 - i];
        x[i + 8] = point[15 - i];
        x[i + 16] = point[23 - i];
        x[i + 24] = point[31 - i];
    }//high->low (x[0]->x[7])     
    //Unpacking...
    //Group the rows for efficient MixColumns implementation
    for (int i = 0; i < 8; i++) {
        SWAPMOVE(x[i + 8], x[i + 0], MASK32, 32);
        SWAPMOVE(x[i + 24], x[i + 16], MASK32, 32);

        SWAPMOVEBY64(x[i + 16], x[i + 0], MASK64);
        SWAPMOVEBY64(x[i + 24], x[i + 8], MASK64);
    }
    //Seperate bits for S-box
    for (int i = 0; i < 4; i++) {
        SWAPMOVE(x[8 * i + 0], x[8 * i + 1], MASK1, 1);
        SWAPMOVE(x[8 * i + 2], x[8 * i + 3], MASK1, 1);
        SWAPMOVE(x[8 * i + 4], x[8 * i + 5], MASK1, 1);
        SWAPMOVE(x[8 * i + 6], x[8 * i + 7], MASK1, 1);

        SWAPMOVE(x[8 * i + 0], x[8 * i + 2], MASK2, 2);
        SWAPMOVE(x[8 * i + 1], x[8 * i + 3], MASK2, 2);
        SWAPMOVE(x[8 * i + 4], x[8 * i + 6], MASK2, 2);
        SWAPMOVE(x[8 * i + 5], x[8 * i + 7], MASK2, 2);

        SWAPMOVE(x[8 * i + 0], x[8 * i + 4], MASK4, 4);
        SWAPMOVE(x[8 * i + 1], x[8 * i + 5], MASK4, 4);
        SWAPMOVE(x[8 * i + 2], x[8 * i + 6], MASK4, 4);
        SWAPMOVE(x[8 * i + 3], x[8 * i + 7], MASK4, 4);
    }
    //至此仅完成SM4加密及数据形式转换...



    //trick0：避免编译器优化
    __m256i and8 = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x00000000000000FF);
    x[0] = _mm256_and_si256(x[0], and8);
    unsigned int* extract = (unsigned int*)&x[0];
    unsigned int back0 = extract[0];
    //printf("\n\n %ud\n\n", extract[0]);
    return back0;
}

