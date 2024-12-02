aaa=test.py
bbb=*2.cpp
ccc=*.cpp

#this is the first file need to compile
nnn=*.tex
#this is the second file need to compile
ooo=mai.tex
#this is the third file need to compile
ppp=mae.tex

aa=a
aaaa=ga

bb=b
bbbb=gb

cc=c
cccc=gc

zz=z
gzz=gz

zzz=zh
gzzz=g2z
oo=o

ddd=$(aaa) $(bbb)

eee=$(aaa) $(bbb) $(ccc)

$(aa):$(aaa)
	python $^
la:$(aaa)
	g++ $< -o $(aa) -llapacke

$(bb):$(bbb)
	g++ $<  -o $@
lb:$(bbb)
	g++ $<  -o $(bb) -llapacke

$(cc):$(ccc)
	g++ $<  -o $@
lc:$(ccc)
	g++ $< -o $(cc) -llapacke

$(zz):$(ddd)
	g++ $^ -o $@
lz:$(ddd)
	g++ $^ -o $(zz) -llapacke
$(zzz):$(eee)
	g++ $^ -o $@
lzz:$(eee)
	g++ $^ -o $(zzz) -llapacke

$(aaaa):$(aaa)
	pdb3.8 $^

$(bbbb):$(bbb)
	g++ -g  $<  -o $@
$(cccc):$(ccc)
	g++ -g  $<  -o $@
$(gzz):$(ddd)
	g++ -g  $^  -o $@
$(gzzz):$(eee)
	g++ -g  $^  -o $@

aa:$(aa)
	./$<

gaa:$(aaaa)
	gdb ./$<
ta:$(aaaa)
	gdb ./$<
va:$(aaaa)
	valgrind --leak-check=full ./$<

bb:$(bb)
	./$< 
gbb:$(bbbb)
	./$< 
tb: $(bbbb)
	gdb $<
vb:$(bbbb)
	valgrind --leak-check=full ./$<



cc:$(cc)
	./$< 
gcc:$(cccc)
	./$< 
tc: $(cccc)
	gdb $<
vc:$(cccc)
	valgrind --leak-check=full ./$<

$(oo):$(ccc)
	g++ $^ -c 


zz:$(zz)
	./$< 
gzz:$(gzz)
	./$< 
tz:$(gzz)
	gdb  $< 
zzz:$(zzz)
	./$< 
gzzz:$(gzzz)
	./$< 
tzz:$(gzzz)
	gdb  $< 
caa:
	rm -r $(aa)
cga:
	rm -r $(aaaa)
cbb:
	rm -r $(bb)
cgb:
	rm -r $(bbbb)
ccc:
	rm -r $(cc)
cgc:
	rm -r $(cccc)
czz:
	rm -r $(zz)
cgz:
	rm -r $(gzz)
czzz:
	rm -r $(zzz)
cgzz:
	rm -r $(gzzz)

clean:
	rm a b c ga gb gc 

clatex:
	rm -r *.aux  *.fls  *.log  *.fdb_latexmk  *.synctex.gz	*.out

xn:$(nnn) 
	xelatex ./$<
	rm -r *.aux  *.fls  *.log  *.fdb_latexmk  

pn:$(nnn)
	pdflatex ./$<
	rm -r *.aux  *.fls  *.log  *.fdb_latexmk  
xo:$(ooo) 
	xelatex ./$<
	rm -r *.aux  *.fls  *.log  *.fdb_latexmk  

po:$(ooo)
	pdflatex ./$<
	rm -r *.aux  *.fls  *.log  *.fdb_latexmk  
xp:$(ppp) 
	xelatex ./$<
	rm -r *.aux  *.fls  *.log  *.fdb_latexmk  

pp:$(ppp)
	pdflatex ./$<
	rm -r *.aux  *.fls  *.log  *.fdb_latexmk  


