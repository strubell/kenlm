#ifndef LM_NGRAM_QUERY_H
#define LM_NGRAM_QUERY_H

#include "lm/enumerate_vocab.hh"
#include "lm/model.hh"
#include "util/file_piece.hh"
#include "util/usage.hh"

#include <cstdlib>
#include <iostream>
#include <ostream>
#include <istream>
#include <string>
#include <queue>

#include <math.h>

namespace lm {
namespace ngram {

struct BasicPrint {
  void Word(StringPiece, WordIndex, const FullScoreReturn &) const {}
  void Line(uint64_t oov, float total) const {
    std::cout << "Total: " << total << " OOV: " << oov << '\n';
  }
  void Summary(double, double, uint64_t, uint64_t) {}
  
};

struct FullPrint : public BasicPrint {
  void Word(StringPiece surface, WordIndex vocab, const FullScoreReturn &ret) const {
    std::cout << surface << '=' << vocab << ' ' << static_cast<unsigned int>(ret.ngram_length)  << ' ' << ret.prob << '\t';
  }

  void Summary(double ppl_including_oov, double ppl_excluding_oov, uint64_t corpus_oov, uint64_t corpus_tokens) {
    std::cout << 
      "Perplexity including OOVs:\t" << ppl_including_oov << "\n"
      "Perplexity excluding OOVs:\t" << ppl_excluding_oov << "\n"
      "OOVs:\t" << corpus_oov << "\n"
      "Tokens:\t" << corpus_tokens << '\n'
      ;
  }
};

struct ProbPair
{
//    lm::WordIndex wordIndex;
    int vocabIndex;
    double probability;

    bool operator<(const ProbPair& t)const
    {
        return probability < t.probability;
    }
};

template <class Model, class Printer> void Query(const Model &model, bool sentence_context)
{
    Printer printer;
    typename Model::State state, out;
    lm::FullScoreReturn ret;
    StringPiece word;

    util::FilePiece in(0);
    srand (time(NULL));

    double corpus_total = 0.0;
    double corpus_total_oov_only = 0.0;
    uint64_t corpus_oov = 0;
    uint64_t corpus_tokens = 0;

    std::string outSent = "";

    // read in our vocab file because fuck figuring this out
    std::vector<StringPiece> generationVocab;
    while (in.ReadWordSameLine(word))
    {
        generationVocab.push_back(word);
        try {
            UTIL_THROW_IF('\n' != in.get(), util::Exception, "FilePiece is confused.");
        } catch (const util::EndOfFileException &e) { break;}
    }

    state = sentence_context ? model.BeginSentenceState() : model.NullContextState();

    int poop = 0;
    // while we are still generating words for current sentence
    // TODO stopping condition
    while (poop < 20)
    {
        ++poop;
        std::priority_queue<ProbPair> probabilityHeap;

        double probSum = 0.0;
        // iterate over each word in our vocabulary and get the probability of choosing it
        for(int i = 0; i < generationVocab.size(); i++)
        {
            word = generationVocab[i];
            ProbPair wordScore;

            lm::WordIndex wordIndex = model.GetVocabulary().Index(word);
            ret = model.FullScore(state, wordIndex, out);

            // store word probabilities as we go
            wordScore.probability = exp(ret.prob);
            wordScore.vocabIndex = i;
            probabilityHeap.push(wordScore);
            probSum += wordScore.probability;

        } // end for over vocab

        std::string chosenWord;
        ProbPair p;
        // limit to top k
        probSum = 0;
        int k = 0;
        // how many top words to choose from
        int K = 250;
        std::vector<ProbPair> topK;
        while (!probabilityHeap.empty() && k < K) {
           p = probabilityHeap.top();
           probSum += p.probability;
           k++; 
           probabilityHeap.pop();
           topK.push_back(p);
        }
        double randPick = static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/probSum));
        double curSum = 0.0;
        // choose a word
        k = 0;
        while (k < topK.size() && curSum < randPick) {
//            p = probabilityHeap.top();
            p = topK[k];
            curSum += p.probability;
            if (curSum > randPick){
                chosenWord = generationVocab[p.vocabIndex].as_string();
                std::cout << p.probability << " : " << chosenWord << std::endl;
            }
//            probabilityHeap.pop();
            k++;
        }
        // update state with chosen word
        lm::WordIndex wordIndex = model.GetVocabulary().Index(chosenWord);
        ret = model.FullScore(state, wordIndex, out);
        outSent += " " + chosenWord;
        state = out;

    } // end while choosing words
    std::cout << outSent << std::endl;

    printer.Summary(
        pow(10.0, -(corpus_total / static_cast<double>(corpus_tokens))), // PPL including OOVs
        pow(10.0, -((corpus_total - corpus_total_oov_only) / static_cast<double>(corpus_tokens - corpus_oov))), // PPL excluding OOVs
        corpus_oov,
        corpus_tokens);
    }

    template <class Model> void Query(const char *file, const Config &config, bool sentence_context, bool show_words) {
    Model model(file, config);
    if (show_words) {
        Query<Model, FullPrint>(model, sentence_context);
    } else {
        Query<Model, BasicPrint>(model, sentence_context);
    }

}

} // namespace ngram
} // namespace lm

#endif // LM_NGRAM_QUERY_H


