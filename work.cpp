#include <time.h>
#include <stdio.h>
#include <math.h>

int main()
{   
    float a;
    time_t t = time(NULL);
    printf("%s", ctime(&t));
    for(int i=0;i<100000;i++){
        for (int j=0;j<10000;j++){
            a = exp(-(i-j)*(i-j));
        }
    }
    time_t t1 = time(NULL);
    printf("%s", ctime(&t1));
}