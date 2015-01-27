//package semilardemo;

import java.io.*;
import java.util.*;
import semilar.config.ConfigManager;
import semilar.data.Sentence;
import semilar.sentencemetrics.BLEUComparer;
import semilar.sentencemetrics.CorleyMihalceaComparer;
import semilar.sentencemetrics.DependencyComparer;
import semilar.sentencemetrics.GreedyComparer;
import semilar.sentencemetrics.LSAComparer;
import semilar.sentencemetrics.LexicalOverlapComparer;
import semilar.sentencemetrics.MeteorComparer;
import semilar.sentencemetrics.OptimumComparer;
import semilar.sentencemetrics.PairwiseComparer.NormalizeType;
import semilar.sentencemetrics.PairwiseComparer.WordWeightType;
import semilar.tools.preprocessing.SentencePreprocessor;
import semilar.tools.semantic.WordNetSimilarity;
import semilar.wordmetrics.LDAWordMetric;
import semilar.wordmetrics.LSAWordMetric;
import semilar.wordmetrics.WNWordMetric;

//Modification of example taken from semilar webpage
//output format SVM 
public class Sentence2SentenceSimilarityTest {

    /* NOTE:
     * The greedy matching and Optimal matching methods rely on word to word similarity method.
     *(please see http://aclweb.org/anthology//W/W12/W12-2018.pdf for more details). So, based on the unique word to
     * word similarity measure, they have varying output (literally, many sentence to sentence similarity methods from
     * the combinations).
     */
    //greedy matching (see the available word 2 word similarity in the separate example file). Here I use some of them
    // for the illustration.
    GreedyComparer greedyComparerWNLin; //greedy matching, use wordnet LIN method for Word 2 Word similarity
    GreedyComparer greedyComparerWNLeskTanim;//greedy matching, use wordnet LESK-Tanim method for Word 2 Word similarity
    GreedyComparer greedyComparerLSATasa; // use LSA based word 2 word similarity (using TASA corpus LSA model).
    GreedyComparer greedyComparerLDATasa; // use LDA based word 2 word similarity (using TASA corpus LDA model).
    //Overall optimum matching method.. you may try all possible word to word similarity measures. Here I show some.
    OptimumComparer optimumComparerWNLin;
    OptimumComparer optimumComparerWNLeskTanim;
    OptimumComparer optimumComparerLSATasa;
    OptimumComparer optimumComparerLDATasa;
    //dependency based method.. we need to provide a word to word similarity metric. Here is just one example
    // using Wordnet Lesk Tanim.
    DependencyComparer dependencyComparerWnLeskTanim;
    //Please see paper Corley, C. and Mihalcea, R. (2005). Measuring the semantic similarity of texts.
    CorleyMihalceaComparer cmComparer;
    //METEOR method (introduced for machine translation evaluation): http://www.cs.cmu.edu/~alavie/METEOR/
    MeteorComparer meteorComparer;
    //BLEU (introduced for machine translation evaluation):http://acl.ldc.upenn.edu/P/P02/P02-1040.pdf 
    BLEUComparer bleuComparer;
    LSAComparer lsaComparer;
    LexicalOverlapComparer lexicalOverlapComparer; // Just see the lexical overlap.
    //For LDA based method.. see the separate example file. Its something different.

    public Sentence2SentenceSimilarityTest() {

        /* Word to word similarity expanded to sentence to sentence .. so we need word metrics */
        boolean wnFirstSenseOnly = false; //applies for WN based methods only.
        WNWordMetric wnMetricLin = new WNWordMetric(WordNetSimilarity.WNSimMeasure.LIN, wnFirstSenseOnly);
        WNWordMetric wnMetricLeskTanim = new WNWordMetric(WordNetSimilarity.WNSimMeasure.LESK_TANIM, wnFirstSenseOnly);
        //provide the LSA model name you want to use.
        LSAWordMetric lsaMetricTasa = new LSAWordMetric("LSA-MODEL-TASA-LEMMATIZED-DIM300");
        //provide the LDA model name you want to use.
        //LDAWordMetric ldaMetricTasa = new LDAWordMetric("LDA-MODEL-TASA-LEMMATIZED-TOPIC300");

        greedyComparerWNLin = new GreedyComparer(wnMetricLin, 0.3f, false);
        greedyComparerWNLeskTanim = new GreedyComparer(wnMetricLeskTanim, 0.3f, false);
        //greedyComparerLSATasa = new GreedyComparer(lsaMetricTasa, 0.3f, false);
        //greedyComparerLDATasa = new GreedyComparer(ldaMetricTasa, 0.3f, false);

        optimumComparerWNLin = new OptimumComparer(wnMetricLin, 0.3f, false, WordWeightType.NONE, NormalizeType.AVERAGE);
        //optimumComparerWNLeskTanim = new OptimumComparer(wnMetricLeskTanim, 0.3f, false, WordWeightType.NONE, NormalizeType.AVERAGE);
        optimumComparerLSATasa = new OptimumComparer(lsaMetricTasa, 0.3f, false, WordWeightType.NONE, NormalizeType.AVERAGE);
        //optimumComparerLDATasa = new OptimumComparer(ldaMetricTasa, 0.3f, false, WordWeightType.NONE, NormalizeType.AVERAGE);

        //Use one of the many word metrics. The example below uses Wordnet Lesk Tanim. Similarly, try using other
        //word similarity metrics.
        dependencyComparerWnLeskTanim = new DependencyComparer(wnMetricLeskTanim, 0.3f, true, "NONE", "AVERAGE");

        /* methods without using word metrics */
        cmComparer = new CorleyMihalceaComparer(0.3f, false, "NONE", "par");
        //for METEOR, please provide the **Absolute** path to your project home folder (without / at the end), And the
        // semilar library jar file should be in your project home folder.
        //meteorComparer = new MeteorComparer("C:/Users/Rajendra/workspace/SemilarLib/");
        bleuComparer = new BLEUComparer();

        //lsaComparer: This is different from lsaMetricTasa, as this method will
        // directly calculate sentence level similarity whereas  lsaMetricTasa
        // is a word 2 word similarity metric used with Optimum and Greedy methods.
        lsaComparer = new LSAComparer("LSA-MODEL-TASA-LEMMATIZED-DIM300");
        lexicalOverlapComparer = new LexicalOverlapComparer(false);  // use base form of words? - No/false. 
        //for LDA based method.. please see the different example file.
    }

    public void printSimilarities(Double score, Sentence sentenceA, Sentence sentenceB, BufferedWriter output) {
        /*System.out.println("Sentence 1:" + sentenceA.getRawForm());
        System.out.println("Sentence 2:" + sentenceB.getRawForm());
        System.out.println("------------------------------");
        System.out.println("greedyComparerWNLin : " + greedyComparerWNLin.computeSimilarity(sentenceA, sentenceB));
        //System.out.println("greedyComparerWNLeskTanim : " + greedyComparerWNLeskTanim.computeSimilarity(sentenceA, sentenceB));
        //System.out.println("greedyComparerLSATasa : " + greedyComparerLSATasa.computeSimilarity(sentenceA, sentenceB));
        //System.out.println("greedyComparerLDATasa : " + greedyComparerLDATasa.computeSimilarity(sentenceA, sentenceB));
        //System.out.println("optimumComparerWNLin : " + optimumComparerWNLin.computeSimilarity(sentenceA, sentenceB));
        //System.out.println("optimumComparerWNLeskTanim : " + optimumComparerWNLeskTanim.computeSimilarity(sentenceA, sentenceB));
        System.out.println("optimumComparerLSATasa : " + optimumComparerLSATasa.computeSimilarity(sentenceA, sentenceB));
        //System.out.println("optimumComparerLDATasa : " + optimumComparerLDATasa.computeSimilarity(sentenceA, sentenceB));
        System.out.println("dependencyComparerWnLeskTanim : " + dependencyComparerWnLeskTanim.computeSimilarity(sentenceA, sentenceB));
        System.out.println("cmComparer : " + cmComparer.computeSimilarity(sentenceA, sentenceB));
        //System.out.println("meteorComparer : " + meteorComparer.computeSimilarity(sentenceA, sentenceB));
        System.out.println("bleuComparer : " + bleuComparer.computeSimilarity(sentenceA, sentenceB));
        System.out.println("lsaComparer : " + lsaComparer.computeSimilarity(sentenceA, sentenceB));
        //System.out.println("lexicalOverlapComparer : " + lexicalOverlapComparer.computeSimilarity(sentenceA, sentenceB));
        System.out.println("                              ");*/
		try{
			output.write(score
						+"\t"+"1:"+lexicalOverlapComparer.computeSimilarity(sentenceA, sentenceB)
						+"\t"+"2:"+greedyComparerWNLin.computeSimilarity(sentenceA, sentenceB)
						+"\t"+"3:"+optimumComparerWNLin.computeSimilarity(sentenceA, sentenceB)
						+"\t"+"4:"+dependencyComparerWnLeskTanim.computeSimilarity(sentenceA, sentenceB)
						+"\t"+"5:"+cmComparer.computeSimilarity(sentenceA, sentenceB)
						+"\t"+"6:"+bleuComparer.computeSimilarity(sentenceA, sentenceB)
						+"\t"+"7:"+lsaComparer.computeSimilarity(sentenceA, sentenceB));
			output.newLine();
			output.flush();
        }catch ( IOException e ){
	       System.err.println("Error: " + e.getMessage());
           e.printStackTrace();
        }
    }

	public List readGS(String file){
		List<Double> gs = new ArrayList();
		try{
			BufferedReader br = new BufferedReader(new FileReader(file));
				
			String line;	
			while ((line = br.readLine()) != null) {
	   			// process the line.
				gs.add(new Double(line));
			}
			br.close();
		}catch(Exception e){//Catch exception if any
			System.err.println("Error: " + e.getMessage());
		}		
		return gs;
	}

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
		if(args.length != 4){
			System.out.println("usage: Sentence2SentenceSimilarityTest <input> <gs> <data-path> <output>");
			System.out.println("usage example: java -cp lib/:Semilar-1.0.jar:. Sentence2SentenceSimilarityTest /media/raid-vapnik/mrios/workspace/STS/datasets/2012/train/STS.input.MSRpar.txt /media/raid-vapnik/mrios/workspace/STS/datasets/2012/train/STS.gs.MSRpar.txt /media/raid-vapnik/mrios/workspace/semilar/data/ test.txt");
			System.exit(0);		
		}
		String input = args[0];
		String gsfile = args[1];
		String path = args[2];
		String out = args[3];

        // first of all set the semilar data folder path (ending with /).
        ConfigManager.setSemilarDataRootFolder(path);

        Sentence sentence1;
        Sentence sentence2;

        /*String text1 = "\"Senator Clinton should be ashamed of herself for playing politics with the important issue of homeland security funding,\" he said.";
        String text2 = "\"She should be ashamed of herself for playing politics with this important issue,\" said state budget division spokesman Andrew Rush.";*/
		BufferedWriter output = null;
		try{
			Sentence2SentenceSimilarityTest s2sSimilarityMeasurer = new Sentence2SentenceSimilarityTest();
			BufferedReader br = new BufferedReader(new FileReader(input));
			String line;
			File file = new File(out);
 
			// if file doesnt exists, then create it
			if (!file.exists()) {
				file.createNewFile();
			} 
			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			output = new BufferedWriter(fw);
			List<Double> gs = s2sSimilarityMeasurer.readGS(gsfile);
			Iterator<Double> igs= gs.iterator();
			int i = 0;
			while ((line = br.readLine()) != null) {
	   			// process the line.
				String[] array = line.split("\\t");
				SentencePreprocessor preprocessor = new SentencePreprocessor(SentencePreprocessor.TokenizerType.STANFORD,
	SentencePreprocessor.TaggerType.STANFORD, SentencePreprocessor.StemmerType.PORTER, SentencePreprocessor.ParserType.STANFORD);
		    	sentence1 = preprocessor.preprocessSentence(array[0]);
		    	sentence2 = preprocessor.preprocessSentence(array[1]);
				System.out.println(array[0] + "|||" + array[1]);
		    	//igs.next();
				Double score =igs.next();
		    	s2sSimilarityMeasurer.printSimilarities(score, sentence1, sentence2, output);
				i++;
			}
			br.close();
			output.close();
		}catch(Exception e){//Catch exception if any
			System.err.println("Error: " + e.getMessage());
		}
		
        System.out.println("\nDone!");
    }
}
