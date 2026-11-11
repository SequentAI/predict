CXX = g++
# no fused multiply-adds, so that scores are rounded alike on every
# architecture and in every method
CXXFLAGS = -O3 -std=c++17 -Wall -ffp-contract=off

SOURCES = main.cpp format.cpp predictor.cpp tfidf.cpp knn.cpp mepo.cpp cosine.cpp \
          nbayes.cpp rforest.cpp dtree.cpp

predict: $(SOURCES)
	$(CXX) $(CXXFLAGS) main.cpp -o $@

# the tests also run on a build that checks memory accesses
predict-checked: $(SOURCES)
	$(CXX) -O1 -g -std=c++17 -Wall -ffp-contract=off -D_GLIBCXX_ASSERTIONS \
	  -fsanitize=address,undefined -fno-sanitize-recover=all main.cpp -o $@

test: predict predict-checked
	tests/run.sh ./predict
	tests/run.sh ./predict-checked

golden: predict
	tests/run.sh ./predict --update-golden

clean:
	rm -f predict predict-checked

.PHONY: test golden clean
