##
## Made by fabien le mentec <texane@gmail.com>
## 
## Started on  Sun Oct 25 10:42:55 2009 texane
## Last update Wed Nov 11 16:49:08 2009 texane
##



CC	:=	gcc
LD	:=	gcc
CFLAGS	:=	-Wall -D_DEBUG=0 -I../src
LFLAGS	:=	
RM	:=	rm -f


BIN	:=	serial

SRCS	:=	../src/main.c		\
		../src/serial.c

OBJS	:=	$(SRCS:.c=.o)


all	:	$(BIN)
$(BIN)	:	$(OBJS)
		$(LD) $(LFLAGS) -o $(BIN) $(OBJS)
clean	:
		$(RM) $(OBJS)
fclean	:	clean
		$(RM) $(BIN)
re	:	fclean all
%.o	:	%.c
		$(CC) $(CFLAGS) -o $@ -c $<
