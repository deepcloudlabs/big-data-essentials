package com.example.service;

import java.io.File;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author Dr. Binnur Kurt <binnur.kurt@gmail.com>
 */
public class BookService {

    private static ServerSocket serverSocket;
    private static final int port = 7070;

    public static void main(String[] args) throws Exception {
        serverSocket = new ServerSocket(port);
        File dictionary = new File("resources", "war-and-peace.txt");
        List<String> words = Files.readAllLines(dictionary.toPath()).stream().map(s -> s.replaceAll("[',!-;\\.\"?]", "")).filter(s -> !s.isEmpty()).collect(Collectors.toList());
        System.err.println("Server is running at port " + port);

        while (true) {
            Socket socket = serverSocket.accept();
            OutputStream outputStream = socket.getOutputStream();
            for (String word : words) {
                System.out.println("Sent: "+word);
                outputStream.write((word + " ").getBytes());
            }
            socket.close();
        }

    }

}
