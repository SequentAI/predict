CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall

# In a clone, the commit comes from git; a source archive has it in main.cpp.
# Only this directory's own .git counts, not that of a repository around it.
COMMIT := $(if $(wildcard .git),$(shell git rev-parse HEAD))
COMMIT_FLAG := $(if $(COMMIT),-DPREDICT_COMMIT='"$(COMMIT)"')

SOURCES = main.cpp format.cpp predictor.cpp tfidf.cpp knn.cpp mepo.cpp \
          mepo_cosine.cpp cosine.cpp nbayes.cpp mesh.cpp rforest.cpp dtree.cpp \
          snapshot.cpp scope.cpp view.cpp versions.cpp

predict: $(SOURCES)
	$(CXX) $(CXXFLAGS) $(COMMIT_FLAG) main.cpp -o $@

# the tests also run on a build that checks memory accesses
predict-checked: $(SOURCES)
	$(CXX) -O1 -g -std=c++17 -Wall -D_GLIBCXX_ASSERTIONS $(COMMIT_FLAG) \
	  -fsanitize=address,undefined -fno-sanitize-recover=all main.cpp -o $@

test: predict predict-checked
	tests/run.sh ./predict
	tests/run.sh ./predict-checked

golden: predict
	tests/run.sh ./predict --update-golden

clean:
	rm -f predict predict-checked

.PHONY: test golden clean
