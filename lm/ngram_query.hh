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

    double corpus_total = 0.0;
    double corpus_total_oov_only = 0.0;
    uint64_t corpus_oov = 0;
    uint64_t corpus_tokens = 0;

    // read in our vocab file because fuck figuring this out
    std::vector<StringPiece> generationVocab;
    while (in.ReadWordSameLine(word))
    {
        generationVocab.push_back(word);
        try {
            UTIL_THROW_IF('\n' != in.get(), util::Exception, "FilePiece is confused.");
        } catch (const util::EndOfFileException &e) { break;}
    }


    // while we are still generating words for current sentence
    // TODO stopping condition
//    while (true)
//    {
        std::priority_queue<ProbPair> probabilityHeap;
        float total = 0.0;
        uint64_t oov = 0;
        double probSum = 0.0;
        // iterate over each word in our vocabulary and get the probability of choosing it
        for(int i = 0; i < generationVocab.size(); i++)
        {
            state = sentence_context ? model.BeginSentenceState() : model.NullContextState();
            word = generationVocab[i];
            ProbPair wordScore;

            lm::WordIndex wordIndex = model.GetVocabulary().Index(word);
            ret = model.FullScore(state, wordIndex, out);
            probSum += ret.prob;
            // store word probabilities as we go
            wordScore.probability = ret.prob;
            wordScore.vocabIndex = i;
//            wordScore.wordIndex = wordIndex;
            probabilityHeap.push(wordScore);

            if (wordIndex == model.GetVocabulary().NotFound()) {
                ++oov;
                corpus_total_oov_only += ret.prob;
            }
            total += ret.prob;
            printer.Word(word, wordIndex, ret);
            ++corpus_tokens;
            state = out;

            if (sentence_context) {
                ret = model.FullScore(state, model.GetVocabulary().EndSentence(), out);
                total += ret.prob;
                ++corpus_tokens;
                printer.Word("</s>", model.GetVocabulary().EndSentence(), ret);
            }

            printer.Line(oov, total);
            corpus_total += total;
            corpus_oov += oov;
        } // end for over vocab

        double randPick = rand() * probSum;
        double curSum = 0.0;
        while (!probabilityHeap.empty()) {
            ProbPair p = probabilityHeap.top();
            if (curSum + p.probability > randPick){
                std::cout << exp(p.probability) + " : " + generationVocab[p.vocabIndex] << std::endl;
            }
            else {
                curSum += p.probability;
            }
            probabilityHeap.pop();
        }
//    } // end while choosing words


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


