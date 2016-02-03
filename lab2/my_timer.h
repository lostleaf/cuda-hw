#include <sys/time.h>
#include <string>
#include <iostream>

#ifndef __MY_TIMER_H
class Timer
{
private:
    timeval t1, t2;
    std::string timer_name;
public:
    Timer(std::string name) : timer_name(name) {}
    void start()
    {
        gettimeofday(&t1, 0);
    }

    void end()
    {
        gettimeofday(&t2, 0);
    }

    void print_elapsed()
    {
        std::cout << timer_name << " time elapsed in microsecond: " << 
            (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec) << std::endl;
    }
};
#define __MY_TIMER_H
#endif
