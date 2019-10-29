
<template>
    <div>
        <ul>
            <li v-for="i in items">
                {{i}}
            </li>
        </ul>
    </div>
</template>


<script>
    define(['Vue'], function (Vue) {
        return new Vue({
            template: template,
            data: {
                items: ['item1', 'item2']
            }
        });
    });
</script>