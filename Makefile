CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall

SOURCES = main.cpp format.cpp predictor.cpp tfidf.cpp knn.cpp mepo.cpp \
          nbayes.cpp rforest.cpp dtree.cpp

predict: $(SOURCES)
	$(CXX) $(CXXFLAGS) main.cpp -o $@

test: predict
	tests/run.sh ./predict

golden: predict
	tests/run.sh ./predict --update-golden

clean:
	rm -f predict

.PHONY: test golden clean
