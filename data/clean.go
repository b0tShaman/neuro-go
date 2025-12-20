package data

import (
	"fmt"
	"os"
	"regexp"
	"strings"
)

var (
	// 1. The Global List: Define all special characters you want to support here.
	// We use strings to support multi-byte characters (like emojis) easily.
	AllowedSpecialChars = []string{
		".", "!", "?", ",", "-", "<", ">", "🙂",
	}

	// 2. Define which of the above characters end a sentence.
	EndTokenChars = []string{".", "!", "?"}

	// 3. Global Regex variables (compiled once at startup)
	ReClean *regexp.Regexp
	ReTok   *regexp.Regexp

	// 4. Global map for fast lookup of end tokens
	EndTokens map[string]bool
)

func init() {
	// --- A. Setup EndTokens Map ---
	EndTokens = make(map[string]bool)
	for _, char := range EndTokenChars {
		EndTokens[char] = true
	}

	// --- B. Build Regex Pattern Components ---

	// Escape all characters to ensure they don't break regex syntax (e.g., "." becomes "\.")
	escapedChars := make([]string, len(AllowedSpecialChars))
	for i, c := range AllowedSpecialChars {
		escapedChars[i] = regexp.QuoteMeta(c)
	}

	// Create a character class for all allowed symbols: e.g., "\.\!\?\,\-\<\>\🙂"
	allAllowedGroup := strings.Join(escapedChars, "")

	// --- C. Compile ReClean ---
	// Logic: Match anything that is NOT a-z, 0-9, or one of our allowed symbols
	// Original: [^a-z0-9.!?,<>-]+
	cleanPattern := fmt.Sprintf(`[^a-z0-9%s]+`, allAllowedGroup)
	ReClean = regexp.MustCompile(cleanPattern)

	// --- D. Compile ReTok ---
	// Logic: match <tags> OR words OR any of the allowed special characters
	// We specifically want to keep the <tag> logic separate so <unk> parses correctly.

	// 1. Tag Pattern: <[^>\s]+>
	tagPattern := `<[^>\s]+>`

	// 2. Word Pattern: [a-z0-9]+
	wordPattern := `[a-z0-9]+`

	// 3. Symbol Pattern: Match any single char from our allowed list
	// Original: [.!?,]|🙂
	symbolPattern := fmt.Sprintf(`[%s]`, allAllowedGroup)

	// Combine: tags | words | symbols
	finalTokPattern := fmt.Sprintf(`%s|%s|%s`, tagPattern, wordPattern, symbolPattern)
	ReTok = regexp.MustCompile(finalTokPattern)
}

func convertLowFreqsToUnk(freqThreshold int, text string, freq map[string]int) string {
	// USE GLOBAL ReTok
	words := ReTok.FindAllString(text, -1)

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
	// 2. Clean text
	// ---------------------------------------------------------
	// A. Replace real line breaks with a special token FIRST
	// text = strings.ReplaceAll(text, "\r\n", " <crlf> ") // Handle Windows endings
	// text = strings.ReplaceAll(text, "\n", " <crlf> ")   // Handle Unix endings
	// text = strings.ReplaceAll(text, "\r", " <crlf> ")   // Handle old Mac endings
	// Optional: You might still want to remove apostrophes specifically if they aren't in your Allowed list
	text = strings.ReplaceAll(text, "'", "")

	// USE GLOBAL ReClean
	text = ReClean.ReplaceAllString(text, " ")
	text = strings.Join(strings.Fields(text), " ")

	os.WriteFile(outputDir+"cleaned_data.txt", []byte(text), 0644)

	// ---------------------------------------------------------
	// 3. Tokenize once to compute frequency
	// ---------------------------------------------------------
	// USE GLOBAL ReTok
	rawWords := ReTok.FindAllString(text, -1)

	freq := make(map[string]int)
	for _, w := range rawWords {
		freq[w]++
	}

	// ---------------------------------------------------------
	// 4. Convert LOW-FREQUENCY words to <unk>
	// ---------------------------------------------------------
	text = convertLowFreqsToUnk(freqThreshold, text, freq)
	os.WriteFile(outputDir+"cleaned_with_unk.txt", []byte(text), 0644)

	// Tokenize AGAIN
	words := ReTok.FindAllString(text, -1)

	// ---------------------------------------------------------
	// 5. Build vocab
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
	var sentence []string

	// Helper to write data
	flushSentence := func(wordsInSentence []string) {
		for i := 0; i < len(wordsInSentence); i++ {
			target := wordsInSentence[i]

			if target == "<unk>" {
				continue
			}

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
					window[p] = pad
					padCount++
				}
			}

			if padCount == contextLen {
				continue
			}

			full := append(window, target)
			fmt.Fprintln(wordDatasetFile, strings.Join(full, ","))

			idRow := make([]string, len(full))
			for j, w := range full {
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
		// USE GLOBAL EndTokens MAP
		if EndTokens[w] {
			sentence = append(sentence, w)
			flushSentence(sentence)
			sentence = []string{}
			continue
		}
		sentence = append(sentence, w)
	}

	if len(sentence) > 0 {
		flushSentence(sentence)
	}

	return vocabSize, wordToID, idToWord
}
