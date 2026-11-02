# Record to work labelling rubric

The base rubric (below the line), applied to an incoming record (`record_a`, one record as it arrived) and a candidate work
(`record_b`, the work as it stood before the record arrived). The rulings above the line were added as cases came up; each
gives the date it was settled. Raw labels are kept; which labels join a work is decided outside the rubric (mergeable =
`same` + `version_preprint`).

**Another journal's publication is its own work** (settled 2026-10-15 after a review of support requests; replaces
the 10-10 rule that a duplicate publication in another journal is `same`, and narrows the 10-13 translation rule): an
article that a different journal (its own ISSN / title) published under its own DOI is `version_other` against the
other journal's article, in any language: a duplicate or co-publication (PRISMA in *BMJ* and *PLOS Medicine*, a trade
piece syndicated in *Bankmagazin* and *Versicherungsmagazin*), a translation in another journal (*ATZ* ↔ *ATZworldwide*,
a Russian original ↔ its English translation journal, a 1949 *Chicago Law Review* article ↔ its Portuguese translation
in *Mises*), and a journal's separately registered language edition (*Anales de Pediatría* ↔ its *English Edition*).
Not another journal, so still `same`: the same journal under a new DOI prefix or a new name, a repository, CRIS,
aggregator or catalogue copy (an aggregator in `venue`, e.g. DOAJ or LA Referencia, is not a journal), and the same
item in another language within one journal or repository: a translated, bilingual or machine-translated title /
abstract (JST, NICT), SciELO's per-language DOIs, the same paper deposited in two languages. A preprint in another
language is still `version_preprint`. A new edition stays `version_other`. This overrides "a translation" under
`version_other` below. A rework that changes the content (restructured, new results) is still `version_other` or
`different`.

**A conference version stays a separate work** (settled 2026-10-13): a meeting or supplement abstract (e.g. *J Hypertens*
44 Suppl 1 e109) or a conference paper is `version_other` against the full journal article, even in the same journal
with the same title and authors. Two copies of the same abstract (a repository copy of the supplement record) are still
`same`.

**A correction, retraction or expression-of-concern notice is `different` from its article** (settled 2026-10-13), even
when the notice repeats the article's title and authors exactly. Signs: Cureus DOIs `10.7759/cureus.c<n>` (correction),
`r<n>` (retraction), `x<n>` (expression of concern); a Crossref title array whose later element is "Correction" /
"Retraction" / "Expression of Concern" (our record keeps only the first element, so the notice can look like the
article); a changed author order with no other difference.

**A book chapter and an article are `different`** (settled 2026-10-13), even with the same title and author: an article
reprinted as a chapter in an edited book (00-33). An article and a chapter are never merged. **A preprint of a chapter is `version_preprint`** (settled
2026-10-15; replaces the 10-13 clause): an author's preprint or accepted manuscript of the chapter on SSRN,
arXiv or a repository joins the published chapter, as any preprint joins its article (01-39). A preprint of a conference paper,
report, book or thesis is likewise `version_preprint` against it. **When the same paper also has a journal article, the preprint
joins the version it declares (arXiv DOI / `IsVersionOf`, Crossref `is-preprint-of`), else the journal article** (settled 2026-10-17); against
the other version it is then `version_other`. The judge record carries no declaration, so the declaration is a model feature (as the nightly's declared-version rule binds).

**Matching trusts the record's type** (settled 2026-10-15; replaces the 10-13 clause "judge what the items are, not the
`type` field"): each record's `type` is taken as given, and a wrong type is fixed in the type system, not at merge. A
repository copy of a chapter typed `article` (00-26) is a typing bug, not a merge case. The boundaries are a fixed veto
outside the judge (`data/training/type_veto.csv`: article-family vs conference paper / abstract, dissertation, report, book chapter,
dataset / software / supplement; book vs chapter or review; notices and peer reviews); vetoed pairs are never judged or
scored. Type pairs no ruling covers (e.g. preprint vs conference paper or report, anything vs `other`) are judged as before.

**Human labels record the truth** (settled 2026-10-13): a human reviewer may open the DOI / landing page, and the gold label
is what that shows, not what the two records alone allow. Such labels carry `outside_evidence: true` in the human labels file
(01-12: a Newcastle deposit "Scottish Celtic Music" = Behr's Bloomsbury entry "Celtic Music in Scotland"). Opus still
judges from the two records only.

**Templated data series stay separate works** (settled 2026-10-13): each upload or release DOI in a templated data series
(EMSL `10.25582/data.<yyyy-mm>.<upload_id>/<osti_id>`, "Data for EMSL Project N from <Month YYYY>") is its own work,
even with an identical title, creators and month. Two such DOIs are `different`, never `same` or `unsure`.
The same holds for any templated series or catalogue (settled 2026-10-13, from human labels 00-10, 02-41, 01-45, 02-44):
sibling photographs of different subjects, municipal project records for different works, and checklist / dataset
records with different keys or regions are `different`. Likewise the same study presented at two meetings (two
conference abstracts with different issue dates, e.g. 02-44: April vs June 2014) is two works. Two records are one work only when they are the same item, not neighbours in a series.

**Declared versions are one work** (settled 2026-10-13: "Declared versions are all one work"; as
the declared-version merges already in OpenAlex): records that declare themselves versions of one item are
`same`, even when the content changes between releases: a DataCite concept DOI and its version DOIs (IsVersionOf /
HasVersion), version to version as well as concept to version; Zenodo version DOIs; `.vN` DOIs; successive releases of one living dataset or software under one concept DOI. The templated-series rule above
covers only undeclared neighbours (EMSL uploads, each its own upload_id with no common concept). New editions, standard
revisions, conference versions and chapter reprints stay separate.
A declaration never joins items of different types (settled 2026-10-15): a dataset, software or other deposit declared a
version of an article (a Zenodo replication package declared `IsVersionOf` the paper) is `different`, as a part of a work.
**One deposit is one work whatever its versions' types** (settled 2026-10-17): a concept DOI and its own versions (same registrant)
are `same` even when the depositor re-typed a version (v1 `article`, v2 `dataset`); the type rule is for declarations between different items.
Crossref-declared versions are one work too (settled 2026-10-15): F1000Research v1/v2, Research Square `/v1` `/v2`, eLife
reviewed-preprint versions. **Peer reviews are `different`** from the item they review (settled 2026-10-15): Crossref
`peer-review` records, referee reports. **Cochrane review updates are `same`** (settled 2026-10-15): `CD…pubN` updates declare
Crossmark `new_version`; a Crossmark `new_edition` stays separate.
Also settled 2026-10-15: multi-part articles (Part I / Part II) are `different`; print and ebook ISBNs of one edition and a
reissue of the same book by another publisher are `same` (book ↔ book; the journal rule above does not apply to books); a retracted article and its
corrected republication under a new DOI (retract-and-replace) are `different`.

---


Adapted from an earlier duplicate-review rubric (2026-09-26), aligned to the 2026-09-28 policy (a preprint and its
published article are one work; conference, thesis and report versions are not).

For each pair of OpenAlex work records, decide whether they should be ONE OpenAlex work.

- `same`: one work, two records: a repository / aggregator / catalogue / legacy (MAG) copy of the same article, chapter,
  book, thesis, dataset or paper; the same item typed differently; a Zenodo/DataCite concept DOI and its version DOI for
  the same item; the same text deposited twice.
- `version_preprint`: a preprint, working paper, discussion paper (e.g. ACPD → ACP), submitted or accepted manuscript and
  the published article of the same paper.
- `version_other`: related but distinct versions that stay separate works: a conference abstract/paper and the later
  journal article, a thesis/dissertation and an article derived from it, a technical report and the article, a new
  edition, a translation.
- `different`: distinct works that share a title or metadata: different items in a series or serial, a book vs its
  chapter or its review, an article vs its supplement / figure / dataset, errata/comments/replies, different papers by the
  same authors, generic titles.
- `unsure`: the records do not carry enough to decide.

Mergeable = `same` + `version_preprint`. Judge only from the two records. Titles, years, authors, venue, biblio, DOI,
URLs and abstract snippets are evidence; a missing field is not evidence of difference; a year gap of 1–2 is normal for
preprint → article and for repository copies.

## Judging notes (v2, 2026-10-06; from the first calibration round: they spell out how the blind labellers already ruled)

- **Catalogue and archive items** (photographs and scans, newspaper or serial issues, herbarium/museum specimens and
  objects, inscriptions, data files inside a dataset, front matter, generic section titles such as "Editorial", "Book
  reviews", "Ressenyes"): an identical title and creator is NOT evidence that two records are one item; collections hold
  many items titled alike. Label `same` only when something distinctive matches (the same object/file/call number, the same
  issue date or number, the same distinctive description); `different` when the records show they are sibling items
  (different numbers, ids, dates or descriptions); `unsure` when nothing distinguishes them.
- **Deposit twins**: a Zenodo (or other DataCite) concept DOI and its version DOI for the same upload, typically adjacent
  DOI numbers with identical metadata, are `same`, and so are two version DOIs of one concept (see Declared versions above). Two different files or records inside
  one dataset are `different`.
- **Repository copies**: a repository / CRIS / aggregator record of a published article (published or accepted version) is
  `same`; a submitted or preprint version of it is `version_preprint`. Both are mergeable.
- **Parts of a work**: an article's supplement, figure, table or dataset; a chapter vs its book; a summary vs the full
  report: `different`.
