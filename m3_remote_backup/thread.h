#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define LENGTH(arr) (sizeof(arr) / sizeof(arr[0]))
#define NumberOfThreads 4

enum {
  T_FREE = 0, // This slot is not used yet.
  T_LIVE,     // This thread is running.
  T_DEAD,     // This thread has terminated.
};

struct thread {
  int id;             // Thread number: 1, 2, ...
  int status;         // Thread status: FREE/LIVE/DEAD
  pthread_t thread;   // Thread struct
  void (*entry)(int); // Entry point
};

// You only allow to create a small number of threads.
static struct thread threads_[NumberOfThreads];
static int n_ = 0;

// This is the entry for a created POSIX thread. It "wraps"
// the function call of entry(id) to be compatible to the
// pthread library's requirements: a thread takes a void *
// pointer as argument, and returns a pointer.
static inline void *wrapper_(void *arg) {
  struct thread *t = (struct thread *)arg;
  t->entry(t->id);
  return NULL;
}

// Create a thread that calls function fn. fn takes an integer
// thread id as input argument.
static inline void create(void *fn) {
  assert(n_ < LENGTH(threads_));

  // Yes, we have resource leak here!
  threads_[n_] = (struct thread){
      .id = n_ + 1,
      .status = T_LIVE,
      .entry = fn,
  };
  pthread_create(&(threads_[n_].thread), // a pthread_t
                 NULL,                   // options; all to default
                 wrapper_,               // the wrapper function
                 &threads_[n_]           // the argument to the wrapper
  );
  n_++;
}

// Wait until all threads return.
static inline void join() {
  for (int i = 0; i < LENGTH(threads_); i++) {
    struct thread *t = &threads_[i];
    if (t->status == T_LIVE) {
      pthread_join(t->thread, NULL);
      t->status = T_DEAD;
    }
  }
}

static inline void resetThreads() {
  for (int i = 0; i < n_; i++) {
    if (threads_[i].status == T_DEAD) {
      threads_[i].status = T_FREE;
    }
  }

  n_ = 0;
}

__attribute__((constructor)) static void startup() { atexit(join); }
