set(BOT_SOURCES
  human/human_bot.cc
  human/human_bot.h
)

add_library(bots OBJECT ${BOT_SOURCES})

add_subdirectory(human)
