all: CUDA mview AV 
#mview_deps = $(shell echo include/mview_{format,interface}.hpp src/mview_format.cpp AbstractViewer/{plottable.{c,h}pp,{viewer_template,shaderprog}.cpp} mview.mk)
#viewer_deps = $(shell echo AbstractViewer/{plottable.{c,h}pp,{viewer_template,shaderprog}.{c,h}pp,Makefile})
mview: CUDA #${mview_deps}
	$(MAKE) -f mview.mk
AV: #${viewer_deps}
	$(MAKE) -C AbstractViewer
CUDA:
	$(MAKE) -f cuda.mk
clean:
	$(MAKE) -f mview.mk clean
	$(MAKE) -C AbstractViewer clean
	$(MAKE) -f cuda.mk clean
