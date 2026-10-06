"""
Parse DailyMed SPL labels into one row per (label NDC, establishment operation).

Row grain is the label's own products, left-joined to the establishment
operations performed on them. A label with no establishment section still
contributes its NDCs, and every operation is kept -- manufacture, repack,
relabel, analysis, pack, label, sterilize -- identified by `opr_code`/`opr_type`.

The XML nests each establishment as its own <assignedEntity>, holding a
DUNS-bearing <assignedOrganization> and that site's own <performance> blocks:

    <author>/<assignedEntity>/<representedOrganization>      <- labeler / registrant
        <assignedEntity>/<assignedOrganization>
            <assignedEntity>                                 <- establishment #1
                <assignedOrganization><id extension="677605851" .../>
                <performance>/<actDefinition>                <- its own operations + NDCs
            <assignedEntity>                                 <- establishment #2
                <assignedOrganization><id extension="915628612" .../>
                <performance>/<actDefinition>

So the DUNS -> NDC link has to be read per operation. Taking the first DUNS in
the document and pairing it with every NDC in the document loses every
establishment after the first and invents links for the ones it keeps.

Columns
-------
zip_name           source zip
ndc                the label's product NDC (5-4), the join key
package_ndc        package NDCs under that product (5-4-2), ";"-joined
name               establishment performing the operation (any operation, see opr_code)
id_duns            establishment DUNS
opr_code           operation code, e.g. C43360
opr_type           operation name, e.g. manufacture
opr_ndc            NDC the operation itself named (differs from `ndc` only when
                   the operation cites a product the label does not describe)
link_type          how the operation was linked to `ndc`: operation | label |
                   none | operation_only | inherited_from_source |
                   source_not_found  (see _LINK_TYPES)
FEI                FDA Establishment Identifier for `id_duns`, from the FDA
                   drug establishment registration file (see add_fei); empty
                   when the DUNS is not currently registered
has_source_ndc     True when the label cites a source NDC for any of its products
source_ndc         the product's source NDC(s), ";"-joined: the product a
                   repackager/relabeler started from, from
                   <asEquivalentEntity><code code="C64637"/><definingMaterialKind>
source_zip_name    for source_ndc rows, the label the inherited operation came from

Source NDCs
-----------
A repackaged product's label usually lists only the repackager's own site, but
cites the product it was packaged from:

    <manufacturedProduct><code code="60687-155"/>               <- label NDC
        <asEquivalentEntity><code code="C64637"/>               <- "source"
            <definingMaterialKind><code code="68382-758"/>      <- source NDC

add_source_operations copies the establishment operations of the source NDC
(read from the source's own label, wherever it is in the corpus) onto the
repackaged NDC as extra rows with link_type = "inherited_from_source". Those
rows keep
`ndc` as the repackaged NDC, set `opr_ndc` to the NDC the operation was
actually recorded for, and `source_zip_name` to the label it came from.
Chains (a relabel of a repack) are followed to the original manufacturer.
"""

import os
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from multiprocessing import Pool

NS = {"ns": "urn:hl7-org:v3"}
_V3 = "{urn:hl7-org:v3}"

DUNS_ROOT = "1.3.6.1.4.1.519.1"       # id/@root that marks a DUNS number
NDC_CODESYSTEM = "2.16.840.1.113883.6.69"

# paths are anchored to this file: <Data>/17 - NDC-FEI Linkage/code/
LINKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.dirname(LINKAGE_DIR)

# FDA Drug Establishments Current Registration Site export (tab-separated)
DRLS_PATH = os.path.join(DATA_DIR, "05 - Firm Level", "drls_reg.txt")

# one zip per SPL label (the full DailyMed human Rx/OTC archive, ~50K zips);
# override with a folder path as the first command-line argument
LABELS_DIR = os.path.join(LINKAGE_DIR, "raw", "all_daily_med_zip")

# the one table this module produces
OUTPUT_PATH = os.path.join(LINKAGE_DIR, "processed", "all_daily_med.csv")

# Set to a collection of operation codes to keep only those, e.g. {"C43360"}
# for manufacture alone. None keeps every operation.
OPERATION_CODES = None

_ORG_TAGS = (f"{_V3}assignedOrganization", f"{_V3}representedOrganization")
_PRODUCT_TAGS = (f"{_V3}manufacturedProduct", f"{_V3}manufacturedMedicine")
_PACKAGE_TAGS = (f"{_V3}containerPackagedProduct", f"{_V3}containerPackagedMedicine")

_LINK_TYPES = {
    "operation": "the operation explicitly named this NDC",
    "label": "operation listed no <product>, so it applies to every product "
             "in the label (the pre-2013 encoding)",
    "none": "no establishment operation covers this NDC",
    "operation_only": "operation named an NDC the label does not describe",
    "inherited_from_source": "operation copied from the label of this NDC's "
                             "source NDC",
    "source_not_found": "the source NDC has no label (or no operations) in "
                        "the corpus",
}

SOURCE_CODE = "C64637"                # asEquivalentEntity code: source product

COLUMNS = ["zip_name", "ndc", "package_ndc", "name", "id_duns",
           "opr_code", "opr_type", "opr_ndc", "link_type", "has_source_ndc",
           "source_ndc"]


def _duns_of(elem):
    """DUNS carried by elem itself or by an organization child of elem."""
    for id_elem in elem.findall(f"./{_V3}id"):
        if id_elem.get("root") == DUNS_ROOT and id_elem.get("extension"):
            return id_elem.get("extension")
    for org in elem:
        if org.tag in _ORG_TAGS:
            for id_elem in org.findall(f"./{_V3}id"):
                if id_elem.get("root") == DUNS_ROOT and id_elem.get("extension"):
                    return id_elem.get("extension")
    return None


def _name_of(elem):
    for org in elem:
        if org.tag in _ORG_TAGS:
            name = org.find(f"./{_V3}name")
            if name is not None and name.text:
                return name.text.strip()
    name = elem.find(f"./{_V3}name")
    return name.text.strip() if name is not None and name.text else None


def _ndc_code(elem):
    code = elem.find(f"./{_V3}code")
    if (code is not None and code.get("code")
            and code.get("codeSystem") == NDC_CODESYSTEM):
        return code.get("code")
    return None


def _source_ndcs(product):
    """Source NDCs a product cites via <asEquivalentEntity code=C64637>."""
    sources = []
    for equiv in product.findall(f"./{_V3}asEquivalentEntity"):
        code = equiv.find(f"./{_V3}code")
        if code is None or code.get("code") != SOURCE_CODE:
            continue
        kind = equiv.find(f"./{_V3}definingMaterialKind")
        ndc = _ndc_code(kind) if kind is not None else None
        if ndc and ndc not in sources:
            sources.append(ndc)
    return sources


def _label_products(root):
    """{product NDC: [package NDCs]} for the products the label describes,
    and {product NDC: [source NDCs]} for those citing a source product."""
    products, sources = {}, {}
    for holder in root.iter(f"{_V3}manufacturedProduct"):
        for product in holder:
            if product.tag not in _PRODUCT_TAGS:
                continue
            ndc = _ndc_code(product)
            if not ndc:
                continue
            for source in _source_ndcs(product):
                if source != ndc and source not in sources.setdefault(ndc, []):
                    sources[ndc].append(source)
            packages = products.setdefault(ndc, [])
            for tag in _PACKAGE_TAGS:
                for package in product.iter(tag):
                    code = _ndc_code(package)
                    if code and code not in packages:
                        packages.append(code)
    return products, sources


def _operations(root):
    """Every establishment operation, with the site that performed it."""
    parents = {child: parent for parent in root.iter() for child in parent}
    operations = []
    for act in root.iter(f"{_V3}actDefinition"):
        code = act.find(f"./{_V3}code")
        opr_code = code.get("code") if code is not None else None
        if OPERATION_CODES is not None and opr_code not in OPERATION_CODES:
            continue

        # nearest enclosing establishment: the first ancestor carrying a DUNS
        duns = name = None
        node = act
        while node in parents:
            node = parents[node]
            duns = _duns_of(node)
            if duns:
                name = _name_of(node)
                break

        operations.append({
            "name": name,
            "id_duns": duns,
            "opr_code": opr_code,
            "opr_type": (code.get("displayName") or "").strip().lower() or None
                        if code is not None else None,
            "ndcs": [c.get("code") for c in act.findall(
                "./ns:product/ns:manufacturedProduct/ns:manufacturedMaterialKind/ns:code",
                NS) if c.get("code")],
        })
    return operations


def parse_fda_xml_content(xml_content, xml_filename):
    """Return a list of dicts, one per (label NDC, establishment operation)."""
    root = ET.fromstring(xml_content)
    products, sources = _label_products(root)
    has_source = bool(sources)

    by_ndc = defaultdict(list)      # label NDC -> operations naming it
    every_product = []              # operations with no <product> of their own
    foreign = []                    # operations naming an NDC the label lacks
    for operation in _operations(root):
        if not operation["ndcs"]:
            every_product.append(operation)
            continue
        for ndc in operation["ndcs"]:
            if ndc in products:
                by_ndc[ndc].append(operation)
            else:
                foreign.append((ndc, operation))

    def row(ndc, package_ndc, operation, opr_ndc, link_type):
        rec = {"zip_name": xml_filename, "ndc": ndc, "package_ndc": package_ndc,
               "opr_ndc": opr_ndc, "link_type": link_type,
               "has_source_ndc": has_source,
               "source_ndc": ";".join(sources.get(ndc, [])) or None}
        for field in ("name", "id_duns", "opr_code", "opr_type"):
            rec[field] = operation[field] if operation else None
        return {field: rec[field] for field in COLUMNS}

    results = []
    for ndc in sorted(products):
        package_ndc = ";".join(sorted(products[ndc])) or None
        covering = by_ndc.get(ndc, []) + every_product
        if not covering:
            results.append(row(ndc, package_ndc, None, None, "none"))
            continue
        for operation in covering:
            # an operation with no <product> applies to every label product
            explicit = ndc in operation["ndcs"]
            results.append(row(ndc, package_ndc, operation,
                               ndc if explicit else None,
                               "operation" if explicit else "label"))

    for ndc, operation in foreign:
        results.append(row(ndc, None, operation, ndc, "operation_only"))

    # keep every label represented, even one with no products and no operations
    if not results:
        results.append(row(None, None, None, None, "none"))

    seen, deduped = set(), []
    for rec in results:
        key = tuple(rec[field] for field in COLUMNS if field != "package_ndc")
        if key not in seen:
            seen.add(key)
            deduped.append(rec)
    return deduped


def _parse_zip(args):
    folder_path, zip_file = args
    try:
        with zipfile.ZipFile(os.path.join(folder_path, zip_file)) as z:
            for xml_name in z.namelist():
                if xml_name.lower().endswith(".xml"):
                    records = parse_fda_xml_content(z.read(xml_name), xml_name)
                    for rec in records:
                        rec["zip_name"] = zip_file
                    return records
    except Exception as exc:                      # keep the label, flag the failure
        rec = {field: None for field in COLUMNS}
        rec["zip_name"] = zip_file
        rec["link_type"] = f"error: {type(exc).__name__}"
        return [rec]
    return []


def process_zip_folder(folder_path, processes=None):
    import pandas as pd

    zip_files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".zip"))
    if not zip_files:
        print("No zip files found.")
        return pd.DataFrame(columns=COLUMNS)

    print(f"Found {len(zip_files)} zip files.")
    step = max(1, len(zip_files) // 10)
    results = []
    with Pool(processes) as pool:
        for idx, records in enumerate(
                pool.imap(_parse_zip, ((folder_path, f) for f in zip_files), chunksize=50),
                start=1):
            results.extend(records)
            if idx % step == 0 or idx == len(zip_files):
                print(f"Progress: {idx / len(zip_files) * 100:.0f}% ({idx}/{len(zip_files)} files processed)")

    return pd.DataFrame(results, columns=COLUMNS)


def add_source_operations(df):
    """Append the source NDC's establishment operations to each NDC citing one.

    Expects `ndc`, `opr_ndc` and `source_ndc` already in the same format
    (see format_ndc). For every (label, NDC) with a source NDC, the operations
    recorded for that source NDC in any label of df are added as rows with
    link_type = "inherited_from_source", opr_ndc = the NDC the operation was
    recorded for, and source_zip_name = that label. A source that is itself
    repackaged is followed further, so a chain reaches the original
    manufacturer. A source with no operations anywhere in df gets one
    "source_not_found" row so the NDC is still visibly flagged.
    """
    import pandas as pd

    op_fields = ["name", "id_duns", "opr_code", "opr_type"]
    base = df[df["link_type"].isin(["operation", "label"])
              & df["ndc"].notna()]
    ops_by_ndc = {
        ndc: group[["zip_name"] + op_fields].drop_duplicates()
        for ndc, group in base.groupby("ndc")
    }

    def split(value):
        return value.split(";") if isinstance(value, str) and value else []

    sources_of = defaultdict(set)   # NDC -> its source NDCs, across all labels
    for ndc, value in df[["ndc", "source_ndc"]].dropna().drop_duplicates() \
            .itertuples(index=False):
        sources_of[ndc].update(split(value))

    targets = (df[df["source_ndc"].notna() & ~df["link_type"].isin(
                   ["operation_only", "inherited_from_source", "source_not_found"])]
               [["zip_name", "ndc", "package_ndc", "has_source_ndc", "source_ndc"]]
               .drop_duplicates(["zip_name", "ndc"]))

    inherited = []
    for target in targets.itertuples(index=False):
        found = False
        queue, seen = split(target.source_ndc), {target.ndc}
        while queue:
            source = queue.pop(0)
            if source in seen:
                continue
            seen.add(source)
            queue.extend(sources_of.get(source, ()))
            ops = ops_by_ndc.get(source)
            if ops is None:
                continue
            found = True
            ops = ops.rename(columns={"zip_name": "source_zip_name"})
            inherited.append(ops.assign(
                zip_name=target.zip_name, ndc=target.ndc,
                package_ndc=target.package_ndc, opr_ndc=source,
                link_type="inherited_from_source", has_source_ndc=target.has_source_ndc,
                source_ndc=target.source_ndc))
        if not found:
            inherited.append(pd.DataFrame([{
                "zip_name": target.zip_name, "ndc": target.ndc,
                "package_ndc": target.package_ndc,
                "link_type": "source_not_found",
                "has_source_ndc": target.has_source_ndc,
                "source_ndc": target.source_ndc}]))

    columns = COLUMNS + ["source_zip_name"]
    out = pd.concat([df] + inherited, ignore_index=True).reindex(columns=columns)
    # keep each label's rows together: its own operations, then inherited ones
    order = {name: i for i, name in enumerate(dict.fromkeys(df["zip_name"]))}
    out["_label"] = out["zip_name"].map(order)
    out["_inherited"] = out["link_type"].isin(["inherited_from_source",
                                               "source_not_found"])
    return (out.sort_values(["_label", "ndc", "_inherited"], kind="stable")
               .drop(columns=["_label", "_inherited"])
               .reset_index(drop=True))


def add_fei(df, drls_path=DRLS_PATH):
    """Left-join the establishment FEI onto df by DUNS (id_duns = DUNS_NUMBER).

    Each DUNS appears once in the registration file, so the row count is
    unchanged; FEI stays a string to keep its leading zeros.
    """
    import pandas as pd

    drls = pd.read_csv(drls_path, sep="\t", dtype=str, index_col=False,
                       encoding="latin-1")
    drls.columns = drls.columns.str.strip()
    fei = (drls[["DUNS_NUMBER", "FEI_NUMBER"]]
           .apply(lambda col: col.str.strip().replace("", None))
           .rename(columns={"DUNS_NUMBER": "id_duns", "FEI_NUMBER": "FEI"})
           .dropna(subset=["id_duns"])
           .drop_duplicates("id_duns"))
    return df.merge(fei, on="id_duns", how="left", validate="many_to_one")


if __name__ == "__main__":
    # regenerate processed/all_daily_med.csv
    import sys
    from ndc_cleaner import format_ndc

    labels_dir = sys.argv[1] if len(sys.argv) > 1 else LABELS_DIR
    if not os.path.isdir(labels_dir):
        sys.exit(f"label folder not found: {labels_dir}")
    df = process_zip_folder(labels_dir)
    for column in ("ndc", "opr_ndc"):
        df = format_ndc(df, column)
    # source_ndc may hold several ";"-joined NDCs; format each one
    sources = df["source_ndc"].str.split(";").explode().to_frame()
    sources = format_ndc(sources.dropna(), "source_ndc")
    df["source_ndc"] = sources.groupby(level=0)["source_ndc"].agg(";".join)
    df = add_source_operations(df)
    df = add_fei(df)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote {OUTPUT_PATH}", df.shape)
