use common::{database, logging, BpResult};
use indicatif::ProgressIterator;
use mongodb::{
    bson::doc,
    sync::{Collection, Database},
};
use structopt::StructOpt;

#[derive(StructOpt)]
struct Cli {
    #[structopt(
        help = "Which collection should the cleaning rules be applied to? Provide \"all\" to clean empirica, ZIMDB_joined and cleaned_80."
    )]
    collection: String,
}

fn main() -> BpResult<()> {
    logging::init_logging();

    let args = Cli::from_args();

    log::info!("Connecting to database...");
    let db = database::get_database(None)?;
    log::info!("Connecting to database... DONE");

    match args.collection.as_ref() {
        "all" => {
            clean_collection(&db, "ZIMDB_joined")?;
            clean_collection(&db, "empirica")?;
            clean_collection(&db, "cleaned_80")?;
        }
        other => {
            let collection_names = db.list_collection_names(None)?;
            if !collection_names.iter().any(|name| name == other) {
                panic!("Could not find collection {} in database.", other);
            }
            clean_collection(&db, other)?
        }
    }

    Ok(())
}

fn clean_collection(db: &Database, collection: &str) -> BpResult<()> {
    let coll = db.collection(collection);

    log::info!("Cleaning collection {}...", collection);
    clean_berlin_kreis(&coll)?;
    clean_munich_kreis(&coll)?;
    clean_wurzburg_kreis(&coll)?;

    let kreis_rewrites: Vec<(&str, &str)> = vec![
        ("Aachen", "Aachen, Städteregion"),
        ("Altenkirchen", "Altenkirchen (Westerwald)"),
        ("Altenkirchen (Ww.)", "Altenkirchen (Westerwald)"),
        ("Amberg", "Amberg-Sulzbach"),
        ("Ansbach (Land)", "Ansbach"),
        ("Aschaffenburg (Land)", "Aschaffenburg"),
        ("Aschaffenburg (Stadt)", "Aschaffenburg, Stadt"),
        ("Augsburgr", "Augsburg"),
        ("Augsburg Land", "Augsburg"),
        ("Augsburg (Land)", "Augsburg"),
        ("Augsburg (Stadt)", "Augsburg, Stadt"),
        ("Bad Tölz - Wolfratshausen", "Bad Tölz-Wolfratshausen"),
        ("Berlin", "Berlin, Stadt"),
        ("Bielefeld", "Bielefeld, Stadt"),
        ("Bamberg (Land)", "Bamberg"),
        ("Bamberg (Stadt)", "Bamberg, Stadt"),
        ("Bayreuth (Land)", "Bayreuth"),
        ("Bayreuth (Stadt)", "Bayreuth, Stadt"),
        ("Berlin Dahlem", "Berlin, Stadt"),
        ("Berlin-Zehlendorf", "Berlin, Stadt"),
        ("Bochum", "Bochum, Stadt"),
        ("Bodensee", "Bodenseekreis"),
        ("Bodenseekreisstarten!", "Bodenseekreis"),
        ("Bonn", "Bonn, Stadt"),
        ("Bonn (Stadt)", "Bonn, Stadt"),
        ("Bottrop", "Bottrop, Stadt"),
        ("Brandenburg a.d.H.", "Brandenburg an der Havel, Stadt"),
        (
            "Brandenburg an der Havel",
            "Brandenburg an der Havel, Stadt",
        ),
        ("Braunschweig", "Braunschweig, Stadt"),
        ("Braunschweig, Kreisfreie Stadt", "Braunschweig, Stadt"),
        ("Bremen", "Bremen, Stadt"),
        ("Bremerhaven - kreisfrei Stadt", "Bremerhaven, Stadt"),
        ("Bremerhaven", "Bremerhaven, Stadt"),
        ("Bördekreis", "Börde"),
        ("Chemnitz (Stadt)", "Chemnitz, Stadt"),
        ("Chemnitz (kreisfreie Stadt)", "Chemnitz, Stadt"),
        ("Chemnitz", "Chemnitz, Stadt"),
        ("Coburg (Land)", "Coburg"),
        ("Coburg (Stadt)", "Coburg, Stadt"),
        ("Cottbus", "Cottbus, Stadt"),
        ("Darmstadt-Dieburg", "Darmstadt, Wissenschaftsstadt"),
        ("Darmstadt", "Darmstadt, Wissenschaftsstadt"),
        ("Delitzsch / Eilenburg", "Delitzsch"),
        ("Delmenhorst", "Delmenhorst, Stadt"),
        ("Dessau", "Dessau-Roßlau, Stadt"),
        ("Dessau-Roßlau", "Dessau-Roßlau, Stadt"),
        ("Dillingen a.d.Donau", "Dillingen a. d. Donau"),
        ("Dortmund", "Dortmund, Stadt"),
        ("Dresden (kreisfreie Stadt)", "Dresden, Stadt"),
        ("Dresden", "Dresden, Stadt"),
        ("Duisburg", "Duisburg, Stadt"),
        ("Stadt Düsseldorf", "Düsseldorf, Stadt"),
        ("Düsseldorf", "Düsseldorf, Stadt"),
        ("Eisenach", "Eisenach, Stadt"),
        ("Emden", "Emden, Stadt"),
        ("Ennepetal", "Ennepe-Ruhr-Kreis"),
        ("Ennepetal-Ruhr-Kreis", "Ennepe-Ruhr-Kreis"),
        ("Erfurt", "Erfurt, Stadt"),
        ("Erlangen", "Erlangen-Höchstadt"),
        ("Essen", "Essen, Stadt"),
        ("Flensburg", "Flensburg, Land"),
        ("Frankenthal", "Frankenthal (Pfalz), kreisfreie Stadt"),
        (
            "Frankenthal (Pfalz)",
            "Frankenthal (Pfalz), kreisfreie Stadt",
        ),
        ("Frankfurt", "Frankfurt am Main, Stadt"),
        ("Frankfurt (Oder)", "Frankfurt (Oder), Stadt"),
        ("Frankfurt am Main", "Frankfurt am Main, Stadt"),
        ("Freiburg", "Freiburg im Breisgau"),
        ("Fürth (Land)", "Fürth"),
        ("Fürth (Stadt)", "Fürth, Stadt"),
        ("Fürther Land", "Fürth"),
        ("Gelsenkirchen", "Gelsenkirchen, Stadt"),
        ("Gera", "Gera, Stadt"),
        ("Hagen", "Hagen, Stadt der FernUniversität"),
        ("Hagen, Stadt", "Hagen, Stadt der FernUniversität"),
        ("Halle (Saale)", "Halle (Saale), Stadt"),
        ("Halle(Saale)", "Halle (Saale), Stadt"),
        ("Hamburg, Hansestadt", "Hamburg, Freie und Hansestadt"),
        ("Hamburg", "Hamburg, Freie und Hansestadt"),
        ("Hamm", "Hamm, Stadt"),
        ("Heilbronn (Land)", "Heilbronn"),
        ("Heilbronn (Stadt)", "Heilbronn, Stadt"),
        ("Heisberg", "Heinsberg"),
        ("Herne", "Herne, Stadt"),
        ("Herzogtum lauenburg", "Herzogtum Lauenburg"),
        ("Hochtaunus", "Hochtaunuskreis"),
        ("Hof (Land)", "Hof"),
        ("Hof (Stadt)", "Hof, Stadt"),
        ("Ilmkreis", "Ilm-Kreis"),
        ("Jena", "Jena, Stadt"),
        ("Kaiserslautern (Land)", "Kaiserslautern"),
        ("Kaiserslautern (Stadt)", "Kaiserslautern, kreisfreie Stadt"),
        ("Karlsruhe (Land)", "Karlsruhe"),
        ("Karlsruhe (Stadt)", "Karlsruhe, Stadt"),
        ("Kassel (Land)", "Kassel"),
        ("Kassel (Stadt)", "Kassel, documenta-Stadt"),
        ("Kehlheim", "Kelheim"),
        ("Kempten", "Kempten (Allgäu)"),
        ("Kiel", "Kiel, Landeshauptstadt"),
        (
            "Kreisfreie Stadt Frankfurt am Main",
            "Frankfurt am Main, Stadt",
        ),
        ("Koblenz, kreisfreie Stadt", "Koblenz, Stadt"),
        ("Koblenz", "Koblenz, Stadt"),
        ("Krefeld", "Krefeld, Stadt"),
        ("Kreis Heinsberg", "Heinsberg"),
        ("Kreis Herzogtum Lauenburg", "Herzogtum Lauenburg"),
        ("Kreis Viersen", "Viersen"),
        ("Kreisfreie Stadt Eisenach", "Eisenach, Stadt"),
        ("Kreisfreie Stadt Gelsenkirchen", "Gelsenkirchen, Stadt"),
        ("Kökn", "Köln, Stadt"),
        ("Köln", "Köln, Stadt"),
        ("Köln (Stadt)", "Köln, Stadt"),
        (
            "Landau in der Pfalz",
            "Landau in der Pfalz, kreisfreie Stadt",
        ),
        ("Landeshauptstadt Magdeburg", "Magdeburg, Landeshauptstadt"),
        ("Landkreis Bautzen", "Bautzen"),
        ("Landkreis Görlitz", "Görlitz"),
        ("Landkreis Leipzig", "Leipzig"),
        ("Landkreis Meißen", "Meißen"),
        ("Landkreis Mittelsachsen", "Mittelsachsen"),
        ("Landkreis München", "München"),
        ("Landkreis Nordsachsen", "Nordsachsen"),
        ("Landkreis Offenbach", "Offenbach"),
        ("Landkreis Rostock", "Rostock"),
        (
            "Landkreis Sächsische Schweiz-Osterzgebirge",
            "Sächsische Schweiz-Osterzgebirge",
        ),
        ("Landkreis Wittenberg", "Wittenberg"),
        ("Landkreis Zwickau", "Zwickau"),
        ("Landsberg a. Lech", "Landsberg am Lech"),
        ("Landshut (Land)", "Landshut"),
        ("Landshut (Stadt)", "Landshut"),
        ("Landshut Land", "Landshut"),
        ("Landshut Stadt", "Landshut, Stadt"),
        ("Landshut (Stadt)", "Landshut, Stadt"),
        ("Leipzig (Stadt)", "Leipzig, Stadt"),
        ("Leipzig (kreisfreie Stadt)", "Leipzig, Stadt"),
        ("Leipzig Land", "Leipzig"),
        ("Leipziger Land", "Leipzig"),
        ("Leipziger- Land", "Leipzig"),
        ("Leverkusen", "Leverkusen, Stadt"),
        ("Lindau", "Lindau (Bodensee)"),
        ("Ludwigshafen", "Ludwigshafen am Rhein, kreisfreie Stadt"),
        (
            "Ludwigshafen am Rhein",
            "Ludwigshafen am Rhein, kreisfreie Stadt",
        ),
        ("Ludwigslust", "Ludwigslust-Parchim"),
        ("Lübeck", "Lübeck, Hansestadt"),
        ("Magdeburg", "Magdeburg, Landeshauptstadt"),
        ("Mainz", "Mainz, kreisfreie Stadt"),
        ("MÖnchenglabdbach", "Mönchengladbach, Stadt"),
        ("MÖnchengladbach", "Mönchengladbach, Stadt"),
        ("Mäönchengladbach", "Mönchengladbach, Stadt"),
        ("Mönchengladbach", "Mönchengladbach, Stadt"),
        ("Mäönchengladbach", "Mönchengladbach, Stadt"),
        ("Mönchengladbach", "Mönchengladbach, Stadt"),
        ("Mönchengladbahc", "Mönchengladbach, Stadt"),
        ("Mönchengnladbach", "Mönchengladbach, Stadt"),
        ("Mönchegladbach", "Mönchengladbach, Stadt"),
        ("Mönchengeladbach", "Mönchengladbach, Stadt"),
        ("Mönchengladbabch", "Mönchengladbach, Stadt"),
        ("Stadt Mönchengladbach", "Mönchengladbach, Stadt"),
        ("Mühldorf a.Inn", "Mühldorf a. Inn"),
        ("Mülheim an der Ruhr", "Mülheim an der Ruhr, Stadt"),
        ("Mülheim", "Mülheim an der Ruhr, Stadt"),
        ("München (Land)", "München"),
        ("München, Stadt", "München, Landeshauptstadt"),
        ("München Stadt", "München, Landeshauptstadt"),
        ("München (Stadt)", "München, Landeshauptstadt"),
        ("München bzw. Ebersberg", "München"),
        ("Münster", "Münster, Stadt"),
        ("Neussf", "Neuss"),
        ("neuss", "Neuss"),
        (
            "Neustadt a.d.Aisch-Bad Windsheim",
            "Neustadt a. d. Aisch - Bad Windsheim",
        ),
        (
            "Neustadt an der Weinstraße",
            "Neustadt an der Weinstraße, kreisfreie Stadt",
        ),
        ("Nienburg", "Nienburg (Weser)"),
        ("Nienburg/Weser", "Nienburg (Weser)"),
        ("Nordsachen", "Nordsachsen"),
        ("Nürnberg Land", "Nürnberg"),
        ("Nürnberger Land", "Nürnberg"),
        ("OberallgäuDr", "Oberallgäu"),
        ("Oberhausen", "Oberhausen, Stadt"),
        ("Offenbach am Main", "Offenbach am Main, Stadt"),
        ("Offenbach", "Offenbach am Main, Stadt"),
        ("Oldenburg (Land)", "Oldenburg"),
        ("Oldenburg (Oldenburg)", "Oldenburg"),
        ("Oldenburg (Oldenburg), Stadt", "Oldenburg, Stadt"),
        ("Osnabrück (Land)", "Osnabrück"),
        ("Osnabrück (Stadt)", "Osnabrück, Stadt"),
        ("Parchim", "Ludwigslust-Parchim"),
        ("Passau (Land)", "Passau"),
        ("Passau (Stadt)", "Passau, Stadt"),
        ("Pfaffenhofen a.d.Ilm", "Pfaffenhofen a. d. Ilm"),
        ("Pirmasens", "Pirmasens, kreisfreie Stadt"),
        ("Potsdam", "Potsdam, Stadt"),
        ("Recklinghauen", "Recklinghausen"),
        ("Regensburg (Land)", "Regensburg"),
        ("Regensburg (Stadt)", "Regensburg, Stadt"),
        ("Remscheid", "Remscheid, Stadt"),
        ("Rhein- Kreis- Neuss", "Rhein Kreis Neuss"),
        ("Rheion-Kreis Neuss", "Rhein Kreis Neuss"),
        ("Rhein-Kreis Neuss", "Rhein Kreis Neuss"),
        ("Rhein-Kreis-Neuss", "Rhein Kreis Neuss"),
        ("Rhein-Erftkreis", "Rhein-Erft-Kreis"),
        ("Rhein-Sieg Kreis", "Rhein-Sieg-Kreis"),
        ("Rhein-Sieg-Keis", "Rhein-Sieg-Kreis"),
        ("Rosenheim (Land)", "Rosenberg"),
        ("Rosenheim (Stadt)", "Rosenheim, Stadt"),
        ("Saale-Holzland", "Saale-Holzland-Kreis"),
        ("Saale-Holzland-kreis", "Saale-Holzland-Kreis"),
        ("Saale-Jolzland-Kreis", "Saale-Holzland-Kreis"),
        ("Saalfeld-Rudolstad", "Saalfeld-Rudolstadt"),
        ("Saalfeld-Rudolstdat", "Saalfeld-Rudolstadt"),
        ("Salzgitter", "Salzgitter, Stadt"),
        ("Saarbrücken", "Stadtverband Saarbrücken"),
        ("Schweinfurt (Land)", "Schweinfurt"),
        ("Schweinfurt (Stadt)", "Schweinfurt, Stadt"),
        ("Solingen, Stadt", "Solingen, Klingenstadt"),
        ("Solingen", "Solingen, Klingenstadt"),
        ("Speyer", "Speyer, kreisfreie Stadt"),
        ("Stadt Brandenburg", "Brandenburg an der Havel, Stadt"),
        (
            "Stadt Brandenburg a.d.H.",
            "Brandenburg an der Havel, Stadt",
        ),
        ("Stadt Chemnitz", "Chemnitz, Stadt"),
        ("Stadt Dessau-Roßlau", "Dessau-Roßlau, Stadt"),
        ("Stadt Halle (Saale)", "Halle (Saale), Stadt"),
        ("Stadt Kevelaer", "Kleve"),
        ("Stuttgart", "Stuttgart, Stadtkreis"),
        ("Straubing", "Straubing-Bogen"),
        ("Suhl", "Suhl, Stadt"),
        ("Sächsische Schweiz", "Sächsische Schweiz-Osterzgebirge"),
        (
            "Sächsische Schweiz - Osterzgebirge",
            "Sächsische Schweiz-Osterzgebirge",
        ),
        ("Trier", "Trier, kreisfreie Stadt"),
        ("UeckerRandow", "Uecker-Randow"),
        ("Vorpommern-Greifswald", "Vorpommern Greifswald"),
        ("Weiden i.d.OPf.", "Weiden i. d. Opf."),
        ("Weimar", "Weimar, Stadt"),
        ("Wiesbaden", "Wiesbaden, Landeshauptstadt"),
        ("Wilhelmshaven", "Wilhelmshaven, Stadt"),
        ("Wolfsburg", "Wolfsburg, Stadt"),
        ("Worms", "Worms, kreisfreie Stadt"),
        ("Wunsiedel i.Fichtelgebirge", "Wunsiedel i. Fichtelgebirge"),
        ("Wuppertal", "Wuppertal, Stadt"),
        ("Würzburg (Land)", "Würzburg"),
        ("Würzburg (Stadt)", "Würzburg, Stadt"),
        ("Würzburg Land", "Würzburg"),
        ("Würzburg Stadt", "Würzburg, Stadt"),
        ("Zweibrücken", "Zweibrücken, kreisfreie Stadt"),
    ];

    log::info!("Rewriting kreise...");
    kreis_rewrites
        .iter()
        .progress()
        .for_each(|(from, to)| rewrite_kreis(&coll, from, to).unwrap());
    log::info!("Rewriting kreise... DONE");

    log::info!("Cleaning collection {}... DONE", collection);

    Ok(())
}

fn rewrite_kreis(coll: &Collection, from: &str, to: &str) -> BpResult<()> {
    coll.update_many(
        doc! {
            "kreis": from,
        },
        doc! {
            "$set": {
                "kreis": to
            }
        },
        None,
    )?;
    Ok(())
}

fn clean_berlin_kreis(coll: &Collection) -> BpResult<()> {
    log::info!("Cleaning Berlin ort, but not kreis...");
    coll.update_many(
        doc! {
            "ort": "Berlin",
            "$or": [
                { "kreis": "Berlin" },
                { "kreis": "" },
                { "kreis": {"$exists": false }}
            ]
        },
        doc! {
            "$set": {
                "kreis": "Berlin, Stadt"
            }
        },
        None,
    )?;
    log::info!("Cleaning Berlin ort, but not kreis... DONE");
    Ok(())
}

fn clean_munich_kreis(coll: &Collection) -> BpResult<()> {
    log::info!("Cleaning München ort, but not kreis...");
    coll.update_many(
        doc! {
            "ort": "München",
            "$or": [
                { "kreis": "München" },
                { "kreis": "München, Stadt" },
                { "kreis": "Berlin, Stadt" }, // This is needed due to a previous bug.
                { "kreis": "Müchen, Landeshauptstadt" }, // This is needed due to a bug in the fix for the previous bug.
                { "kreis": {"$exists": false }}
            ]
        },
        doc! {
            "$set": {
                "kreis": "München, Landeshauptstadt"
            }
        },
        None,
    )?;
    log::info!("Cleaning München ort, but not kreis... DONE");
    Ok(())
}

fn clean_wurzburg_kreis(coll: &Collection) -> BpResult<()> {
    log::info!("Cleaning Würzburg ort, but not kreis...");
    coll.update_many(
        doc! {
            "ort": "Würzburg",
            "$or": [
                { "kreis": "Würzburg" },
                { "kreis": "Würzburg Land" },
                { "kreis": {"$exists": false }}
            ]
        },
        doc! {
            "$set": {
                "kreis": "Würzburg Stadt"
            }
        },
        None,
    )?;
    log::info!("Cleaning Würzburg ort, but not kreis... DONE");
    Ok(())
}
