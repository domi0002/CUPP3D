#include <sys/time.h>

struct Timer 
{
  public:
  double _start_time, _end_time;

  Timer() { }
  ~Timer() { }

  void start() { 
    struct timeval t;    
    gettimeofday(&t, 0); 
    _start_time = t.tv_sec + (1e-6 * t.tv_usec);
  }
  void stop()  { 
    struct timeval t;    
    gettimeofday(&t, 0); 
    _end_time = t.tv_sec + (1e-6 * t.tv_usec);
  }

  float elapsed_sec() { 
    return (float) (_end_time - _start_time);
  }

  float elapsed_ms()
  {
    return (float) 1000 * (_end_time - _start_time);
  }
};



