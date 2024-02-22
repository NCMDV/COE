package com.irefer.service;

import com.irefer.repo.ServiceLinksRepo;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;

import com.irefer.dto.EmailServiceDTO;
import com.irefer.dto.TokenDto;

import org.springframework.web.server.ResponseStatusException;
import reactor.core.publisher.Mono;

import java.util.HashMap;
import java.util.Map;

@Service
@Slf4j
public class EmailService {
    @Value("${spring.token}")
    private String TOKEN_URL;
    @Value("${spring.client.email_service.id}")
    private String CLIENT_ID;
    @Value("${spring.client.email_service.username}")
    private  String CLIENT_USERNAME;
    @Value("${spring.client.email_service.password}")
    private String CLIENT_PASSWORD;
    @Autowired
    ServiceLinksRepo serviceLinksRepo;



    public String SendEmail(EmailServiceDTO request) {
//      String token = getToken().token();
        String BASE_URI = serviceLinksRepo.findByKey("email service").getLink();

        EmailServiceDTO result = WebClient.create(BASE_URI)
                .post()
//		        .header(HttpHeaders.AUTHORIZATION, "Bearer " + token)
                .accept(MediaType.APPLICATION_JSON)
                .body(BodyInserters.fromValue(request))
                .retrieve()
                .bodyToMono(EmailServiceDTO.class).block();

        log.info("Email id: {}", result.getId());
        log.info("Email status: {}", result.getEmailStatus());
        return result.getEmailStatus();
    }

    public String SendSecuredEmail(EmailServiceDTO request) {
        String token = getToken().token();
        String BASE_URI = serviceLinksRepo.findByKey("secured email service").getLink();

        EmailServiceDTO result = WebClient.create(BASE_URI)
                .post()
                .header(HttpHeaders.AUTHORIZATION, "Bearer " + token)
                .accept(MediaType.APPLICATION_JSON)
                .body(BodyInserters.fromValue(request))
                .retrieve()
                .bodyToMono(EmailServiceDTO.class).block();

        log.info("Email id: {}", result.getId());
        log.info("Email status: {}", result.getEmailStatus());
        return result.getEmailStatus();
    }

    public TokenDto getToken() {
        try {
            Map<String, String> cred = new HashMap<>();
            cred.put("email", CLIENT_USERNAME);
            cred.put("password", CLIENT_PASSWORD);

            String EMAIL_SERVICE_URL = serviceLinksRepo.findByKey("email service token").getLink();

            WebClient client = WebClient.builder()
                    .baseUrl(EMAIL_SERVICE_URL)
                    .defaultHeader("Content-Type", MediaType.APPLICATION_JSON_VALUE)
                    .build();

            Mono<TokenDto> responseMono = client.post()
                    .body(BodyInserters.fromValue(cred))
                    .retrieve()
                    .bodyToMono(TokenDto.class);

            TokenDto response = responseMono.block(); // Block to wait for the response

            return response;
        }catch (Exception e) {
            e.printStackTrace();
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Unauthorized: " + e.getMessage(), e);
        }
    }


}
