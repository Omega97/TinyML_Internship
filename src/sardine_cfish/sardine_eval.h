#ifndef SARDINE_EVAL_H
#define SARDINE_EVAL_H

#include "types.h"

struct Position;
Value sardine_evaluate(const Position *pos);
int sardine_init(const char *path);

#endif
