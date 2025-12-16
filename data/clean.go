package data

import (
	"fmt"
	"os"
	"regexp"
	"slices"
	"strings"
)

var endTokens = map[string]bool{
	".":   true,
	"?":   false,
	"!":   false,
	"eos": false,
}

// -------------------------------------------------------------
// Convert low-frequency words to <unk>
// -------------------------------------------------------------
func convertLowFreqsToUnk(freqThreshold int, text string, freq map[string]int) string {
	reTok := regexp.MustCompile(`<[^>\s]+>|[a-z0-9]+|[.!?,]|🙂`)
	words := reTok.FindAllString(text, -1)

	out := make([]string, len(words))
	for i, w := range words {
		if count, ok := freq[w]; ok && count < freqThreshold {
			out[i] = "<unk>"
		} else {
			out[i] = w
		}
	}
	return strings.Join(out, " ")
}

// // -------------------------------------------------------------
// // PreprocessNGramData
// // -------------------------------------------------------------
func PreprocessNGramData(contextLen, freqThreshold int, filepath, outputDir string) (
	vocabSize int,
	wordToID map[string]int,
	idToWord []string,
) {

	// ---------------------------------------------------------
	// 1. Read file
	// ---------------------------------------------------------
	data, err := os.ReadFile(filepath)
	if err != nil {
		panic(err)
	}
	text := strings.ToLower(string(data))

	// ---------------------------------------------------------
	// 2. Clean text (allow < and > so tokens like <unk> survive)
	// ---------------------------------------------------------
	text = strings.ReplaceAll(text, "'", "")

	// include '<' and '>' so <unk> remains intact after cleaning
	re := regexp.MustCompile(`[^a-z0-9.!?,<>-]+`)
	text = re.ReplaceAllString(text, " ")

	text = strings.Join(strings.Fields(text), " ")

	// temporary cleaned text (before unk replacement)
	os.WriteFile(outputDir+"cleaned_data.txt", []byte(text), 0644)

	// ---------------------------------------------------------
	// 3. Tokenize once to compute frequency BEFORE unk replacement
	//    Use tokenizer that will preserve existing <...> tokens if any
	// ---------------------------------------------------------
	reTok := regexp.MustCompile(`<[^>\s]+>|[a-z0-9]+|[.!?,]`)
	rawWords := reTok.FindAllString(text, -1)

	// Build frequency table (counts plain tokens as they appear now)
	freq := make(map[string]int)
	for _, w := range rawWords {
		freq[w]++
	}

	// ---------------------------------------------------------
	// 4. Convert LOW-FREQUENCY words to <unk>
	// ---------------------------------------------------------
	// You can make freqThreshold a parameter if you want
	text = convertLowFreqsToUnk(freqThreshold, text, freq)

	// save after unk replacement
	os.WriteFile(outputDir+"cleaned_with_unk.txt", []byte(text), 0644)

	// Tokenize AGAIN after <unk> conversion (this tokenizer treats <unk> as one token)
	words := reTok.FindAllString(text, -1)

	// ---------------------------------------------------------
	// 5. Build vocab (include <pad> and <unk> explicitly)
	// ---------------------------------------------------------
	wordToID = map[string]int{
		"<pad>": 0,
		"<unk>": 1,
	}
	idToWord = []string{"<pad>", "<unk>"}

	for _, w := range words {
		if _, ok := wordToID[w]; !ok {
			wordToID[w] = len(idToWord)
			idToWord = append(idToWord, w)
		}
	}

	vocabSize = len(idToWord)

	// ---------------------------------------------------------
	// 6. Write mapping
	// ---------------------------------------------------------
	f1, _ := os.Create(outputDir + "mapping.txt")
	defer f1.Close()
	for id, w := range idToWord {
		fmt.Fprintf(f1, "%s - %d\n", w, id)
	}

	// ---------------------------------------------------------
	// 7. Create sliding-window dataset
	// ---------------------------------------------------------
	wordDatasetFile, _ := os.Create(outputDir + "dataset_words.csv")
	idDatasetFile, _ := os.Create(outputDir + "dataset.csv")
	defer wordDatasetFile.Close()
	defer idDatasetFile.Close()

	pad := "<pad>"

	// ---------------------------------------
	// Sentence buffer
	// ---------------------------------------
	var sentence []string

	flushSentence := func(wordsInSentence []string) {

		for i := 0; i < len(wordsInSentence); i++ {
			target := wordsInSentence[i]

			// ------------------------------------------------
			// IMPORTANT RULE: DO NOT predict <unk> as target
			// ------------------------------------------------
			if target == "<unk>" {
				continue
			}

			// build input window with proper indexing
			window := make([]string, contextLen)
			padCount := 0

			for p := 0; p < contextLen; p++ {
				idx := i - contextLen + p
				if idx < 0 {
					window[p] = pad
					padCount++
				} else if idx >= 0 && idx < len(wordsInSentence) {
					window[p] = wordsInSentence[idx]
				} else {
					// out of range (shouldn't normally happen) -> pad
					window[p] = pad
					padCount++
				}
			}

			// skip fully padded row
			if padCount == contextLen {
				continue
			}

			// final row (inputs + target)
			full := append(window, target)

			// write dataset_words.csv
			fmt.Fprintln(wordDatasetFile, strings.Join(full, ","))

			// write dataset.csv (map words to ids)
			idRow := make([]string, len(full))
			for j, w := range full {
				// fallback: if for some reason word not in vocab, use <unk>
				id, ok := wordToID[w]
				if !ok {
					id = wordToID["<unk>"]
				}
				idRow[j] = fmt.Sprintf("%d", id)
			}
			fmt.Fprintln(idDatasetFile, strings.Join(idRow, ","))
		}
	}

	sentence = []string{}

	for _, w := range words {

		if endTokens[w] {
			// include the "." as a final target and flush
			sentence = append(sentence, w)
			flushSentence(sentence)
			sentence = []string{}
			continue
		}

		// otherwise keep accumulating
		sentence = append(sentence, w)
	}

	// flush last sentence if no ending punctuation
	if len(sentence) > 0 {
		flushSentence(sentence)
	}

	return vocabSize, wordToID, idToWord
}

// PreprocessSeqData: final corrected version (same signature)
func PreprocessSeqData(contextLen, freqThreshold int, filepath, outputDir string) (
	vocabSize int,
	wordToID map[string]int,
	idToWord []string,
) {
	// --------------------------
	// 1. Read file
	// --------------------------
	data, err := os.ReadFile(filepath)
	if err != nil {
		panic(err)
	}
	text := strings.ToLower(string(data))

	// --------------------------
	// 2. Clean text
	// --------------------------
	text = strings.ReplaceAll(text, "'", "")
	re := regexp.MustCompile(`[^a-z0-9.!?,<>-\x{1F642}]+`)
	text = re.ReplaceAllString(text, " ")
	text = strings.Join(strings.Fields(text), " ")
	_ = os.WriteFile(outputDir+"cleaned_data.txt", []byte(text), 0644)

	// tokenizer
	reTok := regexp.MustCompile(`<[^>\s]+>|[a-z0-9]+|[.!?,]|🙂`)
	rawWords := reTok.FindAllString(text, -1)

	// --------------------------
	// 3. Frequency table (before unk)
	// --------------------------
	freq := make(map[string]int)
	for _, w := range rawWords {
		freq[w]++
	}

	// --------------------------
	// 4. Replace low freq -> <unk>
	// --------------------------
	text = convertLowFreqsToUnk(freqThreshold, text, freq)
	_ = os.WriteFile(outputDir+"cleaned_with_unk.txt", []byte(text), 0644)
	words := reTok.FindAllString(text, -1)

	// --------------------------
	// 5. Build vocab
	// --------------------------
	wordToID = map[string]int{
		"<pad>": 0,
		"<unk>": 1,
	}
	idToWord = []string{"<pad>", "<unk>"}

	for _, w := range words {
		if _, ok := wordToID[w]; !ok {
			wordToID[w] = len(idToWord)
			idToWord = append(idToWord, w)
		}
	}
	vocabSize = len(idToWord)

	// --------------------------
	// 6. Write mapping
	// --------------------------
	fmap, _ := os.Create(outputDir + "mapping.txt")
	defer fmap.Close()
	for id, w := range idToWord {
		fmt.Fprintf(fmap, "%s - %d\n", w, id)
	}

	// --------------------------
	// 7. Create LSTM windows (CORRECTED)
	// --------------------------
	wordCSV, _ := os.Create(outputDir + "dataset_words.csv")
	idCSV, _ := os.Create(outputDir + "dataset.csv")
	defer wordCSV.Close()
	defer idCSV.Close()

	var sentence []string

	flushSentence := func(s []string) {
		// s is a single sentence including its final punctuation (if any)
		n := len(s)
		if n == 0 {
			return
		}

		L := contextLen
		// start index range: start from -(L-1) (to produce left-padded early windows)
		// last start such that Y's last index == n-1 is start = n-1-L
		start := -(L - 1)
		end := n - 1 - L
		// If end < start, no windows where Y fully fits inside sentence -> return early
		if end < start {
			return
		}

		for st := start; st <= end; st++ {
			// Build X : indices st .. st+L-1
			X := make([]string, L)
			for k := 0; k < L; k++ {
				idx := st + k
				if idx < 0 {
					X[k] = "<pad>"
				} else {
					X[k] = s[idx]
				}
			}

			// Build Y : indices st+1 .. st+L
			Y := make([]string, L)
			skip := false
			allPad := true
			for k := 0; k < L; k++ {
				idx := st + k + 1
				if idx < 0 || idx >= n {
					Y[k] = "<pad>"
				} else {
					Y[k] = s[idx]
					if Y[k] == "<unk>" {
						skip = true
					}
					if Y[k] != "<pad>" {
						allPad = false
					}
				}
			}

			// do not predict <unk>
			if skip {
				continue
			}
			// drop meaningless all-pad Y (shouldn't happen given bounds, but keep safe)
			if allPad {
				continue
			}

			full := slices.Concat(X, Y)
			fmt.Fprintln(wordCSV, strings.Join(full, ","))

			idRow := make([]string, len(full))
			for j, w := range full {
				id := wordToID[w]
				if id == 0 && w != "<pad>" { // fallback: unknown token not in vocab (shouldn't happen)
					id = wordToID["<unk>"]
				}
				idRow[j] = fmt.Sprintf("%d", id)
			}
			fmt.Fprintln(idCSV, strings.Join(idRow, ","))
		}
	}

	// build sentences and flush on punctuation
	sentence = []string{}
	for _, w := range words {
		sentence = append(sentence, w)
		if endTokens[w] {
			flushSentence(sentence)
			sentence = []string{}
		}
	}
	// final partial sentence
	if len(sentence) > 0 {
		flushSentence(sentence)
	}

	return vocabSize, wordToID, idToWord
}
