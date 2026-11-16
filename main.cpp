#include <getopt.h>
#include <iomanip>
#include <functional>
#include <memory>
#include <sstream>

#include "cosine.cpp"
#include "mepo_cosine.cpp"
#include "mesh.cpp"
#include "rforest.cpp"
#include "scope.cpp"
#include "snapshot.cpp"

// Raise with every change to rankings, options or formats.
const string version_str = "0.8.0";

const string usage_str =
  "Usage: predict <syms> <deps> <seq> [OPTION]...\n"
  "\n"
  "Mandatory arguments:\n"
  "  <syms> is a file containing the symbols of theorems\n"
  "  <deps> is a file containing the dependencies of theorems\n"
  "  <seq>  is a file containing the order of theorems\n"
  "\n"
  "General options:\n"
  "  -p <method>  method is either knn, mepo, mepo-cosine, nbayes, cosine, mash,\n"
  "               mesh, or rforest\n"
  "  -n <i>       i is the number of predictions to output\n"
  "  -e <eval>    eval is an optional file, containing theorems for which we want predictions\n"
  "  -S           print each prediction's score after it, in parentheses\n"
  "  -x <path>    eXport learned predictor data to path\n"
  "  -y <path>    Ymport learned predictor data from path\n"
  "  -o <file>    write the corpus to a snapshot file and stop\n"
  "  -i <file>    start from the corpus in a snapshot file; <syms>, <deps> and\n"
  "               <seq> then give theorems to add after it. Exits with status 3\n"
  "               if the snapshot cannot be used\n"
  "  -L <file>    file giving the theories of theorems, for query scopes\n"
  "  -b <x>       in a query scope, the weight of a rank is multiplied by x for\n"
  "               the theorems of the ancestors (default 4)\n"
  "\n"
  "Predictor-specific options:\n"
  " k-NN (also in MaSh and MeSh):\n"
  "  -c <0|1>     1 credits the dependencies of neighbours, 0 does not (default 1)\n"
  " Naive Bayes (also in MaSh and MeSh):\n"
  "  -d <n>       learn a theorem's features for only its first n dependencies\n"
  "               (default 20; 0 for all)\n"
  " MePo (also in MeSh):\n"
  "  -F <file>    file giving the statures (simp, local) of theorems\n"
  "  -P           (mepo only, with queries) match symbols by their types\n"
  " MeSh:\n"
  "  -T <cut>     how MeSh cuts its inputs for M predictions: 2M+25 (default)\n"
  "               cuts MePo to M and MaSh to 2M+25 theorems, 51n/50 cuts both\n"
  "               to 51M/50\n"
  " Random Forest:\n"
  "  -t <n>       number of trees to build\n"
  "  -s <n>       number of samples to consider per tree\n"
  "  -f <n>       number of features to consider per tree\n"
  "  -w <n>       dependency weight\n"
  "\n"
  "Methods nbayes-dense and cosine-dense compute the rankings of nbayes and\n"
  "cosine by scanning every theorem; they are slow references for the tests.\n"
  "\n"
  "predict --version prints the version.\n";


void print_prediction(LDPairVec prediction, long n_predictions,
  vector<string> no_th, bool scores) {
  for (long j = 0; j < n_predictions; ++j) {
    // print label
    cout << no_th[prediction[j].first] << " ";
    // print weight
    if (scores)
      cout << "(" << setprecision(17) << prediction[j].second << ") ";
  }
  cout << endl;
}

// a query scope with ancestors re-ranks this many predictions, or k if more
const long boost_depth = 128;

// The state of the interactive mode: the corpus, and [make], which builds a
// predictor for it.
struct Session {
  Corpus& corpus;
  vector<unsigned char>& statures;
  Theories& theories;
  function<unique_ptr<Predictor>()> make;
  long predno;
  double boost;
  bool scores;
  // with [types], the predictor is a MePo over typed symbols
  TypeTable* types;
  // whether the corpus may change (!add and !replace)
  bool growable;
};

// The four files `<syms> <deps> <seq> <theories>` named in [args], each
// checked to be readable, so that a command fails before it changes the
// corpus.
vector<string> command_files(const string& args, const string& line) {
  istringstream in(args);
  vector<string> files(4);
  string rest;
  if (!(in >> files[0] >> files[1] >> files[2] >> files[3]) || (in >> rest))
    exit_error("Expected four files in `" + line + "'!");
  for (const auto& f : files)
    if (!ifstream(f))
      exit_error("Cannot read `" + f + "' in `" + line + "'!");
  return files;
}

// Adds the theorems of [files], as command_files gives them, after the
// others.
void add_theorems(Session& s, const vector<string>& files) {
  const string &symsf = files[0], &depsf = files[1], &seqf = files[2],
    &theoriesf = files[3];
  Corpus& c = s.corpus;
  read_order(seqf, c.th_no, c.no_th);
  const long n = c.no_th.size();
  c.deps.resize(n);
  c.syms.resize(n);
  s.statures.resize(n, 0);
  s.theories.resize(n);
  read_deps(depsf, c.th_no, c.deps);
  long sym_num = c.no_sym.size();
  read_syms(symsf, c.syms, c.sym_ths, sym_num, c.th_no, c.sym_no, c.no_sym);
  read_theories(theoriesf, c.th_no, s.theories);
}

// Runs the command [line]: `!add <syms> <deps> <seq> <theories>` adds
// theorems after the others, `!replace <theory> <syms> <deps> <seq>
// <theories>` removes the theorems of <theory> first.
void command(Session& s, const string& line) {
  if (!s.growable)
    exit_error("`" + line + "' does not work with -P, -y or -x!");
  if (line.rfind("!add ", 0) == 0)
    add_theorems(s, command_files(line.substr(5), line));
  else if (line.rfind("!replace ", 0) == 0) {
    const size_t space = line.find(' ', 9);
    const string theory = line.substr(9, space - 9);
    if (!valid_theory(theory) || space == string::npos)
      exit_error("Expected `!replace <theory> <syms> <deps> <seq> "
                 "<theories>', not `" + line + "'!");
    const vector<string> files = command_files(line.substr(space + 1), line);
    const long t = s.theories.find(theory);
    if (t != -1) remove_theory(s.corpus, s.statures, s.theories, t);
    add_theorems(s, files);
  }
  else
    exit_error("Unknown command `" + line + "'!");
}

// Answers the queries and runs the commands on standard input, one output
// line for each.
void interaction(unique_ptr<Predictor>& p, Session& s) {
  p->learn_all();
  cerr << "Learning done; awaiting your features ..." << endl;

  string line;
  while (getline(cin, line)) {
    const long n = s.corpus.no_th.size();
    if (!line.empty() && line[0] == '!') {
      command(s, line);
      p = s.make();
      p->learn_all();
      cout << s.corpus.no_th.size() << endl;
      continue;
    }

    Scope scope;
    const size_t start = parse_scope(line, scope);
    const bool scoped = start > 0;
    const long k = min(n, scope.k > 0 ? scope.k : s.predno);
    const long no_adv = scope.ancestors.empty() ? k
      : min(n, max(k, boost_depth));
    const vector<char> in_scope = scoped
      ? scope_mask(scope, s.theories) : vector<char>();
    p->set_scope(scoped ? &in_scope : nullptr);

    LDPairVec ans;
    if (s.types)
      ans = static_cast<MePo&>(*p).predict_typed(
        parse_typed_query(line.begin() + start, line.end(), s.corpus.sym_no,
          *s.types), n, no_adv);
    else {
      LDPairVec query = parse_feature_list(line.begin() + start, line.end(),
        s.corpus.sym_no);
      // after !replace, a feature may have no theorem left
      query.erase(remove_if(query.begin(), query.end(),
        [&](const pair<long, double>& f) {
          return s.corpus.sym_ths[f.first].empty(); }), query.end());
      ans = p->predict_weighted(query, n, no_adv);
    }
    ans.resize(min(no_adv, (long)ans.size()));
    if (!scope.ancestors.empty())
      boost_ancestors(ans, scope, s.theories, s.boost);
    print_prediction(ans, min(k, (long)ans.size()), s.corpus.no_th, s.scores);
  }
}

void evaluation(unique_ptr<Predictor>& p, string evalf, long predno,
  SLMap th_no, vector<string> no_th, bool scores) {
  unordered_set<long> eval;
  read_eval(evalf, th_no, eval);

  // last theorem up to which we learnt
  long prev = 0;

  for (long i = 0; i < (long)no_th.size(); ++i) {
    if (eval.find(i) != eval.end()) {
      p->learn(prev, i);
      prev = i;

      long no_adv = min(i, predno);
      LDPairVec ans = p->predict(i, i, no_adv);

      cout << no_th[i] << ":";
      print_prediction(ans, no_adv, no_th, scores);
    }
  }
}

int atoi_check(const char *nptr, const char *desc) {
  int result = atoi(nptr);
  if (result == 0) {
    cerr << "Error: You have to specify a valid " << desc << "!\n";
    exit(EXIT_FAILURE);
  }
  else
    return result;
}

double atof_check(const char *nptr, const char *desc) {
  double result = atof(nptr);
  if (result == 0.0) {
    cerr << "Error: You have to specify a valid " << desc << "!\n";
    exit(EXIT_FAILURE);
  }
  else
    return result;
}

int main(int argc, char* argv[]) {
  const int MIN_ARGS = 3;
  if (argc == 2 && string(argv[1]) == "--version") {
    cout << "predict " << version_str << endl;
    return EXIT_SUCCESS;
  }
  if (argc < MIN_ARGS + 1) {
    cerr << usage_str;
    return EXIT_FAILURE;
  }

  // obligatory files
  string symsf(argv[1]), depsf(argv[2]), seqf(argv[3]);

  // evaluation file
  string evalf = "";
  // number of predictions to output
  long predno = 1000;
  // prediction method
  string method = "knn";
  // path to prelearned predictor data
  string import_path, export_path;
  // snapshot files to start from and to write
  string snapshot_in, snapshot_out;

  // Random Forest specific options
  long n_trees = 512;
  long n_samples = 512;
  long n_features = 128;
  double depweight = 1.7;

  // Naive Bayes specific options
  long max_deps = 20;

  // MePo specific options
  string staturesf;
  bool typed_mepo = false;

  // k-NN specific options
  bool credit_deps = true;

  // MeSh specific options
  bool over = false;

  // query scopes
  string theoriesf;
  double boost = 4;

  // print scores
  bool scores = false;

  int c;
  optind = MIN_ARGS + 1;  // start getopt after obligatory arguments
  while ((c = getopt (argc, argv, "hb:c:d:e:F:i:L:n:o:p:Pk:St:T:s:f:w:x:y:")) != -1)
    switch (c) {
      case 'h':
        cout << usage_str;
        return EXIT_SUCCESS;
      case 'c':
        if (string(optarg) != "0" && string(optarg) != "1") {
          cerr << "Error: -c takes 0 or 1, not '" << optarg << "'!\n";
          return EXIT_FAILURE;
        }
        credit_deps = string(optarg) == "1";
        break;
      case 'T':
        if (string(optarg) != "2M+25" && string(optarg) != "51n/50") {
          cerr << "Error: -T takes 2M+25 or 51n/50, not '" << optarg << "'!\n";
          return EXIT_FAILURE;
        }
        over = string(optarg) == "51n/50";
        break;
      case 'd': {
        char *rest;
        errno = 0;
        max_deps = strtol(optarg, &rest, 10);
        if (*optarg == '\0' || *rest != '\0' || errno != 0 || max_deps < 0) {
          cerr << "Error: You have to specify a valid number of dependencies!\n";
          return EXIT_FAILURE;
        }
        break;
      }
      case 'e':
        evalf = optarg;
        break;
      case 'F':
        staturesf = optarg;
        break;
      case 'L':
        theoriesf = optarg;
        break;
      case 'b': {
        char *rest;
        boost = strtod(optarg, &rest);
        if (*optarg == '\0' || *rest != '\0' || !isfinite(boost) ||
            boost <= 0) {
          cerr << "Error: -b takes a positive number, not '" << optarg
               << "'!\n";
          return EXIT_FAILURE;
        }
        break;
      }
      case 'i':
        snapshot_in = optarg;
        break;
      case 'o':
        snapshot_out = optarg;
        break;
      case 'n':
        predno = atoi_check(optarg, "number of predictions");
        break;
      case 'p':
        method = optarg;
        break;
      case 'P':
        typed_mepo = true;
        break;
      case 'S':
        scores = true;
        break;
      case 'x':
        export_path = optarg;
        break;
      case 'y':
        import_path = optarg;
        break;
      case 't':
        n_trees = atoi_check(optarg, "number of trees");
        break;
      case 's':
        n_samples = atoi_check(optarg, "number of samples per tree");
        break;
      case 'f':
        n_features = atoi_check(optarg, "number of features per tree");
        break;
      case 'w':
        depweight = atof_check(optarg, "dependency weight");
        break;
      case '?':
        // unknown option or option lacking an argument
        // getopt prints an error message (unless opterr is set to 0)
        cerr << "Try '" << argv[0] << " -h' for more information.\n";
        return EXIT_FAILURE;
    }

  if (typed_mepo && (method != "mepo" || !evalf.empty())) {
    cerr << "Error: -P works only with -p mepo, and not with -e!\n";
    return EXIT_FAILURE;
  }
  if (typed_mepo && !(snapshot_in.empty() && snapshot_out.empty())) {
    cerr << "Error: -P does not work with snapshots!\n";
    return EXIT_FAILURE;
  }
  // exported trees name their features, which a snapshot does not hold
  if (!export_path.empty() && !snapshot_in.empty()) {
    cerr << "Error: -x does not work with -i!\n";
    return EXIT_FAILURE;
  }

  long sym_num = 0;
  SLMap th_no;           // maps a theorem to its numeric identifier
  HLMap sym_no;          // maps a symbol's hash to its numeric identifier
  vector<string> no_th,  // theorem name table
                 no_sym;
  LVecVec deps,          // dependencies of each theorem
          syms,          // syms[t] holds the symbols of a theorem t
          sym_ths;       // sym_ths[s] holds the theorems which contain s

  Corpus corpus{no_th, no_sym, th_no, sym_no, deps, syms, sym_ths};
  if (!snapshot_in.empty()) {
    const string why = read_snapshot(snapshot_in, version_str, corpus);
    if (!why.empty()) {
      cerr << "Error: cannot use snapshot `" << snapshot_in << "': " << why
           << "!\n";
      return 3;
    }
    sym_num = no_sym.size();
  }

  read_order(seqf, th_no, no_th);

  deps.resize(no_th.size());
  syms.resize(no_th.size());

  read_deps(depsf, th_no, deps);
  TypeTable types;
  TypedSyms typed(typed_mepo ? no_th.size() : 0),
    freq_only(typed_mepo ? no_th.size() : 0);
  read_syms(symsf, syms, sym_ths, sym_num, th_no, sym_no, no_sym,
    typed_mepo ? &types : nullptr, &typed, &freq_only);

  if (!snapshot_out.empty()) {
    write_snapshot(snapshot_out, version_str, corpus);
    return EXIT_SUCCESS;
  }

  vector<unsigned char> statures(no_th.size(), 0);
  if (!staturesf.empty())
    read_statures(staturesf, th_no, statures);

  // getting number of a theorem
  //cout << th_no["Set.subsetI"] << endl;

  Theories theories;
  theories.resize(no_th.size());
  if (!theoriesf.empty())
    read_theories(theoriesf, th_no, theories);

  // builds a predictor for the corpus as it is
  auto make = [&]() {
    unique_ptr<Predictor> p;
    const long sym_num = no_sym.size();
    if (method == "knn")
      p.reset(new kNN(deps, syms, sym_ths, sym_num, credit_deps));
    else if (method == "mepo")
      p.reset(new MePo(deps, syms, sym_ths, sym_num, statures,
        typed_mepo ? &types : nullptr, typed_mepo ? &typed : nullptr,
        typed_mepo ? &freq_only : nullptr));
    else if (method == "mepo-cosine")
      p.reset(new MePoCosine(deps, syms, sym_ths, sym_num));
    else if (method == "nbayes" || method == "nbayes-dense")
      p.reset(new NaiveBayes(deps, syms, sym_ths, sym_num, max_deps,
        method == "nbayes-dense"));
    else if (method == "cosine" || method == "cosine-dense")
      p.reset(new Cosine(deps, syms, sym_ths, sym_num,
        method == "cosine-dense"));
    else if (method == "mash")
      p.reset(new MaSh(deps, syms, sym_ths, sym_num, max_deps,
        credit_deps));
    else if (method == "mesh")
      p.reset(new MeSh(deps, syms, sym_ths, sym_num, statures, max_deps,
        credit_deps, over));
    else if (method == "rforest")
      p.reset(new RandomForest(deps, syms, sym_ths, sym_num,
        n_trees, n_samples, n_features, depweight));
    else
      return p;
    p->set_tables(no_th, no_sym, th_no, sym_no);
    if (!import_path.empty())
      p->import_data(import_path);
    return p;
  };

  unique_ptr<Predictor> predictor = make();
  if (!predictor) {
    cerr << "Error: You have to specify a valid predictor!\n";
    return EXIT_FAILURE;
  }

  // if user did not supply evaluation file
  if (evalf == "") {
    Session session{corpus, statures, theories, make, predno, boost, scores,
      typed_mepo ? &types : nullptr,
      !typed_mepo && import_path.empty() && export_path.empty()};
    interaction(predictor, session);
  }
  else
    evaluation(predictor, evalf, predno, th_no, no_th, scores);

  if (!export_path.empty())
    predictor->export_data(export_path);
}
