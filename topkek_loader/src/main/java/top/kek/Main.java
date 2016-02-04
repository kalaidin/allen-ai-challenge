package top.kek;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class Main {

    private static List<String> download_kek(String url) {
        try {
            List<String> result = new ArrayList<>();
            Document doc = Jsoup.connect(url).timeout(10000).get();
            for (Element tr : doc.select("table#theTable").select("tr")) {
                StringBuilder sb = new StringBuilder();
                for (Element td : tr.select("td")) {
                    sb.append(td.text()).append('\t');
                }
                result.add(sb.toString());
            }
            return result;
        } catch (IOException e) {
            System.out.println("We fucked up");
            return new ArrayList<>();
        }
    }

    private static String download_dict(String url) {
        try {
            Document doc = Jsoup.connect(url).timeout(10000).get();
            return doc.select("table.tabbed").select("tr").get(1).select("td").get(0).text();
        } catch (IOException e) {
            System.out.println("We fucked up");
            return "";
        }
    }

    public static void parse_everything(String links, String output_path, int n_threads) throws InterruptedIOException, IOException {
        List<String> urls = new ArrayList<>();
        FileReader fr = new FileReader(links);
        BufferedReader br = new BufferedReader(fr);
        String buf = br.readLine();
        while (buf != null) {
            urls.add(buf);
            buf = br.readLine();
        }
        System.out.println(urls.size() + " links");
        br.close();


        long start = System.currentTimeMillis();
        AtomicInteger count = new AtomicInteger(0);
        FileWriter output = new FileWriter(new File(output_path));
        ExecutorService service = Executors.newFixedThreadPool(n_threads);
        List<CompletableFuture<String>> results = new ArrayList<>();
        for (String url : urls) {
            results.add(CompletableFuture.supplyAsync(() -> download_dict(url), service));
        }
        for (CompletableFuture<String> f : results) {
            f.whenComplete((strings, throwable) -> {
                synchronized (br) {
                    try {
                        output.write(strings + "\n");
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                if (count.incrementAndGet() % 100 == 0) {
                    System.out.println(count.get() + " in " + (System.currentTimeMillis() - start) / 1000.0);
                }
            });
        }
    }

    public static void main(String[] args) throws InterruptedException, IOException {
        
    }
}
