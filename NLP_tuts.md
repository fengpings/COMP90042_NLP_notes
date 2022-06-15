# NLP Tuts
## Week 1
- What is tokenisation and why is it important?
	- Tokenisation is the act of transforming a (long) document into a set of meaningful substrings, so that we can compare with other (long) documents.	- In general, a document is too long — and contains too much information — to manipulate directly. There are some counter-examples, like language identification, which we need to perform before we decide how to tokenise anyway.
- What are stemming and lemmatisation, and how are they different?
	- Both stemming and lemmatisation are mechanisms for transforming a token into a canonical (base, normalised) form. For example, turning the token walking into its base form walk.
	- Both operate by applying a series of rewrite operations to remove or replace (parts of) affixes (primarily suffixes). (In English, anyway.)
	- However, **lemmatisation works in conjunction with a lexicon**: a list of valid words in the language. The goal is to turn the input token into an element of this list (a valid word) using the rewrite rules. If the re-write rules can’t be used to transform the token into a valid word, then the token is left alone. (For example, the token lemming wouldn’t be transformed into lemm because the latter isn’t in the word list.)
	- **Stemming simply applies the rewrite rules, even if the output is a garbage token (like lemm).**
	- One further idea is the difference between inflectional morphology and derivational morphology:
		- Inflectional morphology is the systematic process (in many but not all languages) by which tokens are altered to conform to certain grammatical constraints: for example, if the English noun teacher is plural, then it must be represented as teachers. The idea is that these changes don’t really alter the meaning of the term. Consequently, **both stemming and lemmatisation attempt to remove this kind of morphology**.
		- Derivational morphology is the (semi-)systematic process by which we transform terms of one class into a different class. For example, if we would like to make the English verb teach into a noun (someone who performs the action of teaching), then it must be represented as teacher. This kind of morphology tends to produce terms that differ (perhaps subtly) in meaning, and the two separate forms are usually both listed in the lexicon. Consequently, **lemmatisation doesn’t usually remove derivational morphology in its normalisation process, but stemming usually does.**

## Week 2
![](tut/1.png)

## Week 3
- What is a **POS tag**?
	- A POS tag, AKA word classes, is a label assigned to a word token in a sentence which indicates some grammatical (primarily syntactic) properties of its function in the sentence.
- What are some common approaches to POS tagging? What aspects of the data might allow us to predict POS tags systematically?
	- **Unigram**: Assign a POS tag to a token according to the most common observation in a tagged corpus; many words are unambiguous, or almost unambiguous.
	- **N-gram**: Assign a POS tag to a token according to the most common tag in the same sequence (based on the sentence in which the token occurs) of n tokens (or tags) in the tagged corpus; context helps disambiguate.
	- **Rule-based**: Write rules (relying on expertise of the writer) that disambiguate unigram tags.
	- **Sequential**: Learn a Hidden Markov Model (or other model) based on the observed tag sequences in a tagged corpus.
	- **Classifier**: Treat as a supervised machine learning problem, with tags from a tagged corpus as training data.
- What are the **assumptions that go into a Hidden Markov Model**? What is the time complexity of the **Viterbi algorithm**? Is this practical?
	- **Markov assumption**: the likelihood of transitioning into a given state depends only on the current state, and not the previous state(s) (or output(s))
	- **Output independence assumption**: the likelihood of a state producing a certain word (as output) does not depend on the preceding (or following) state(s) (or output(s)).
	- The time complexity of the Viterbi algorithm, for an HMM with T possible states, and a sentence of length W, is O(T2W). In POS tagging, there might typically be approximately 100 possible tags (states), and a typical sentence might have 10 or so tokens, so ... yes, it is practical (unless we need to tag really, really, quickly, e.g. tweets as they are posted).
- how can the **initial state probabilities** $\pi$ be esitmated
	- Record the distribution of tags for the first token of each sentence in a tagged corpus.
- How can the **transition probabilities** A be estimated?
	- For each tag, record the distribution of tags of the immediately following token in the tagged corpus. (We might need to introduce an end–of–sentence dummy for the probabilities to add up correctly.)
- How can the **emission probabilities** B be estimated?
	- For each tag, record the distribution of corresponding tokens in the tagged corpus.
![](tut/2.png) ![](tut/3.png) ![](tut/4.png) ![](tut/5.png)
